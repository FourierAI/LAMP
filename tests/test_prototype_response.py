import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import contextlib
import copy
import io
import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch.nn import functional as F
import main as m
from runtime_checks import execution_environment, record_runtime

torch.set_num_threads(1)
torch.manual_seed(39)
s = {'features': F.normalize(torch.randn(12, 5), dim=-1),
     'native': [F.normalize(torch.randn(12, 4, d), dim=-1) for d in (4, 6, 8)],
     'labels': torch.arange(12) % 3}
q = {'features': F.normalize(torch.randn(7, 5), dim=-1),
     'native': [F.normalize(torch.randn(7, 4, d), dim=-1) for d in (4, 6, 8)]}
text = F.normalize(torch.randn(3, 5), dim=-1).T.contiguous()
beta, lam = .75, .2
rhs = m.residual(s['features'], s['labels'], text)
gram_global = m.global_gram(s['features'], beta)
alpha_global = m.solve(gram_global, rhs, lam)
g_response = m.solve(gram_global, s['features'], lam)
branches, gram_regional = [], []
for native in s['native']:
    gram, state = m.center_support(m.region_gram(native, beta * .5))
    gram_regional.append(gram)
    branches.append(dict(alpha=m.solve(gram, rhs, lam), state=state, beta=beta * .5,
                         response=m.solve(gram, s['features'], lam)))


def expect_error(fn, contains):
    try:
        fn()
    except RuntimeError as error:
        assert contains in str(error), str(error)
    else:
        raise AssertionError('Expected error: ' + contains)


class PrototypeResponseTests(unittest.TestCase):
    def test_active_responses_refits_and_legacy_cache(self):
        with TemporaryDirectory(prefix='lamp_response_regression_') as folder:
            root = Path(folder)
            cases = [(0., m.EQUAL, 1), (.4, m.EQUAL, 4), (.7, (0., .25, .75), 3),
                     (1., (.75, .25, 0.), 2), (1., (0., 0., 1.), 1)]
            for number, (rho, weights, expected_solves) in enumerate(cases):
                output = root / str(number)
                output.mkdir()
                args = Namespace(output=output, device='cpu')
                selection = {'global_parameters': {'beta': beta, 'lambda': lam},
                             'regional_parameters': {'rho': rho, 'weights': list(weights), 'ratio': .5}}
                legacy = dict(alpha=alpha_global, response_global=g_response,
                              branches=copy.deepcopy(branches) if rho else [])
                model = dict(alpha=alpha_global,
                             branches=[{k: v for k, v in branch.items() if k != 'response'} for branch in legacy['branches']])
                m.write(output / 'selection.json', selection)
                torch.save(model, output / 'model.pt')
                seals = (m.sha(output / 'selection.json'), m.sha(output / 'model.pt'))
                expect_error(lambda: m.load_prototype_response(args, s, model, selection), 'run --method both --stage adapt')
                assert not (output / 'prototype_response.pt').exists()
                with patch.object(m, 'solve', wraps=m.solve) as solve:
                    responses = m.prepare_prototype_response(args, s, model, selection)
                    assert solve.call_count == expected_solves, (number, solve.call_count)
                receipt = json.loads((output / 'prototype_response.json').read_text())
                assert receipt['new_solves'] == expected_solves
                assert seals == (m.sha(output / 'selection.json'), m.sha(output / 'model.pt'))
                with patch.object(m, 'solve', side_effect=AssertionError('Unexpected response recomputation')):
                    repeated = m.prepare_prototype_response(args, s, model, selection)
                    loaded = m.load_prototype_response(args, s, model, selection)
                    actual = m.response_scores(q, s, model, selection, loaded)
                old_global = m.global_response(q['features'], s['features'], legacy['response_global'], beta)
                old_terms = [m.native_delta(xq, xs, dict(branch, alpha=branch['response']))
                             for xq, xs, branch in zip(q['native'], s['native'], legacy['branches'])]
                expected = m.mix(torch.zeros_like(old_global), old_global, old_terms, selection['regional_parameters'])
                assert torch.equal(actual, expected), (number, (actual - expected).abs().max())
                assert actual.shape == (len(q['features']), 5)
                if rho == 1: assert responses['global'] is None
                for i, active in enumerate(receipt['active_regional']):
                    assert (responses['regional'][i] is not None) == active
                # Compare the cached prototype update with independent full branch refits.
                gamma = F.normalize(torch.randn(3, 5), dim=-1) - F.normalize(text.T, dim=-1)
                new_rhs = F.one_hot(s['labels'], 3).float() - s['features'] @ (text.T + gamma).T
                new_global = m.global_response(q['features'], s['features'], m.solve(gram_global, new_rhs, lam), beta)
                new_terms = [m.native_delta(xq, xs, dict(branch, alpha=m.solve(gram, new_rhs, lam)))
                             for xq, xs, branch, gram in zip(q['native'], s['native'], model['branches'], gram_regional)]
                direct = m.mix(q['features'] @ (text.T + gamma).T, new_global, new_terms, selection['regional_parameters'])
                free = m.free_scores(q, s, text, model, selection)
                cached = free + q['features'] @ gamma.T - actual @ gamma.T
                torch.testing.assert_close(cached, direct, rtol=2e-5, atol=3e-6)
                assert torch.equal(free + q['features'] @ torch.zeros_like(gamma).T - actual @ torch.zeros_like(gamma).T, free)
                # Legacy embedded responses can be migrated without any extra solves.
                legacy_dir = output / 'legacy'
                legacy_dir.mkdir()
                legacy_args = Namespace(output=legacy_dir, device='cpu')
                m.write(legacy_dir / 'selection.json', selection)
                with patch.object(m, 'solve', side_effect=AssertionError('Legacy response should be reused')):
                    migrated = m.prepare_prototype_response(legacy_args, s, legacy, selection)
                    assert torch.equal(m.response_scores(q, s, legacy, selection, migrated), expected)
                assert json.loads((legacy_dir / 'prototype_response.json').read_text())['new_solves'] == 0
            print('PASS: active branches, rho endpoints, zero-weight depth slots, equal-weight ordering, cache reuse, legacy cache migration, full-refit identity, zero-update identity')


    def test_selection_never_prepares_prototype_responses(self):
        with TemporaryDirectory(prefix="lamp_selection_regression_") as folder:
            root = Path(folder)
            # A tiny full search must fit residuals only, even when both methods are requested.
            output = root / 'selection'; output.mkdir()
            config = root / 'source' / 'configs'; config.mkdir(parents=True)
            (config / 'global_grid.json').write_text(json.dumps({'datasets': {'toy': {'beta': [.5, 1., 2], 'lambda_': [.1, .3, 2], 'candidate_count': 4}}}))
            support = dict(s, bank_sha256='fixture')
            query = dict(q, labels=torch.arange(7) % 3, bank_sha256='fixture')
            for name, value in [('support', support), ('validation', query), ('text', text)]:torch.save(value, output / (name + '.pt'))
            args = Namespace(output=output, device='cpu', dataset='toy', bank_sha256='fixture', selection_evaluation_overlap=False, dense_limit=12000, method='both')
            with patch.object(m, 'HERE', config.parent), patch.object(m, 'solve', wraps=m.solve) as solve, contextlib.redirect_stdout(io.StringIO()):
                m.select(args)
            assert solve.call_count == 4 + 9
            assert all(call.args[1].shape[1] == 3 for call in solve.call_args_list)
            model = torch.load(output / 'model.pt', weights_only=True)
            assert 'response_global' not in model and all('response' not in branch for branch in model['branches'])
            assert not (output / 'prototype_response.pt').exists()
            print('PASS: full 337-candidate selection performs exactly global-grid + 9 residual solves; no prototype response')


    def test_selected_model_binds_selection_and_features(self):
        with TemporaryDirectory(prefix='lamp_selection_seal_') as directory:
            output = Path(directory)
            for name in ('support', 'validation', 'text'):
                torch.save(torch.zeros(2, 2), output / (name + '.pt'))
            selection = dict(feature_sha256={name: m.sha(output / (name + '.pt'))
                                           for name in ('support', 'validation', 'text')},
                             regional_parameters=dict(rho=.5))
            torch.save(dict(selection=copy.deepcopy(selection), branches=[]), output / 'model.pt')
            selection['model_sha256'] = m.sha(output / 'model.pt')
            m.write(output / 'selection.json', selection)
            args = Namespace(output=output, device='cpu')
            m.load_selected_model(args)
            selection['regional_parameters']['rho'] = 1.
            m.write(output / 'selection.json', selection)
            with self.assertRaisesRegex(RuntimeError, 'Selection record differs'):
                m.load_selected_model(args)
            selection['regional_parameters']['rho'] = .5
            m.write(output / 'selection.json', selection)
            torch.save(torch.ones(2, 2), output / 'support.pt')
            with self.assertRaisesRegex(RuntimeError, 'Fit input changed: support'):
                m.load_selected_model(args)


if __name__ == '__main__':
    unittest.main()
