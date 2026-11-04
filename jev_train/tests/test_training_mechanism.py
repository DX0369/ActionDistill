from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))
import analyze_training_mechanism as a


def configuration():
    return dict(world_size=1, accum=2, max_tokens=32768, seed=20261015,
                epochs=1.0, max_steps=0)


class TrainingExposureTests(unittest.TestCase):
    def test_different_batch_sizes_have_different_sample_coefficients(self):
        selected = [dict(id='s1', n_prompt_tokens=100), dict(id='s2', n_prompt_tokens=100),
                    dict(id='long', n_prompt_tokens=30000)]
        exposure, audit = a.loss_exposure(selected, configuration(),
                                         [dict(step=1, total=1, tokens=30200)])
        self.assertEqual(exposure['s1'], dict(occurrences=1, loss_coefficient=0.25))
        self.assertEqual(exposure['s2'], dict(occurrences=1, loss_coefficient=0.25))
        self.assertEqual(exposure['long'], dict(occurrences=1, loss_coefficient=0.5))
        self.assertEqual(audit['loss_coefficient_sum'], 1.0)

    def test_epoch_rounding_consumes_next_epoch_instead_of_dropping_batch(self):
        selected = [dict(id=str(i), n_prompt_tokens=20000) for i in range(3)]
        events = [dict(step=i, total=2, tokens=40000) for i in (1, 2)]
        exposure, audit = a.loss_exposure(selected, configuration(), events)
        self.assertEqual(sorted(r['occurrences'] for r in exposure.values()), [1, 1, 2])
        self.assertEqual(audit['effective_prompt_tokens'], 80000)
        self.assertEqual(audit['final_epoch'], 1)
        self.assertEqual(audit['loss_coefficient_sum'], 2.0)

    def test_ddp_averaging_matches_same_global_microbatch_count(self):
        selected = [dict(id='s1', n_prompt_tokens=100), dict(id='s2', n_prompt_tokens=100),
                    dict(id='long', n_prompt_tokens=30000)]
        events = [dict(step=1, total=1, tokens=30200)]
        c = configuration()
        one = a.loss_exposure(selected, c, events)
        two = a.loss_exposure(selected, dict(c, world_size=2, accum=1), events)
        self.assertEqual(one, two)

    def test_missing_length_and_wrong_logged_tokens_fail(self):
        with self.assertRaises(KeyError):
            a.loss_exposure([dict(id='x')], configuration(), [])
        with self.assertRaisesRegex(ValueError, 'token schedule differs'):
            a.loss_exposure([dict(id='x', n_prompt_tokens=20000)], configuration(),
                            [dict(step=1, total=1, tokens=20000)])


def view(messages):
    capture = dict(messages=messages, context=dict(source='fixture.json', strip_think=False))
    return a.TurnView(dict(capture=capture, family='openai', raw_records=(), record=None, prefix=None))


def call(cid):
    return dict(role='assistant', content=None,
                tool_calls=[dict(id=cid, function=dict(name='run', arguments='{}'))])


class ErrorSemanticsTests(unittest.TestCase):
    def test_text_error_and_exit_code_are_not_guessed(self):
        self.assertEqual(a.explicit_error({}, 'Error: failed, exit code 1'), (False, None))
        self.assertEqual(a.explicit_error({}, '{"exit_code": 1}'), (False, None))
        self.assertEqual(a.explicit_error({}, '{"error": "failed"}'), (True, None))
        self.assertEqual(a.explicit_error({}, '{"error": null}'), (False, None))
        self.assertEqual(a.explicit_error({'is_error': True}, 'ok'), (True, None))
        self.assertEqual(a.explicit_error({'is_error': 'false'}, 'ok')[1], 'invalid_explicit_error_flag')

    def test_current_turn_only_and_no_future_tool_return(self):
        messages = [dict(role='user', content='first'), call('a'),
                    dict(role='tool', tool_call_id='a', content='{"error":"failure"}'),
                    dict(role='assistant', content='done'), dict(role='user', content='next'), call('b'),
                    dict(role='tool', tool_call_id='b', content='ok'),
                    dict(role='assistant', content='done')]
        v = view(messages)
        self.assertEqual(a.previous_error(v, 3, v.prior(3))['status'], 'explicit_error')
        self.assertEqual(a.previous_error(v, 7, v.prior(7))['status'], 'no_explicit_error')
        self.assertEqual(a.previous_error(v, 6, v.prior(6))['status'], 'unknown')

    def test_zero_calls_and_unmatched_returns_are_separate(self):
        v = view([dict(role='user', content='work'), dict(role='assistant', content='reply')])
        self.assertEqual(a.previous_error(v, 1, v.prior(1))['status'], 'no_prior_call')
        v = view([dict(role='user', content='work'), call('a'),
                  dict(role='tool', tool_call_id='wrong', content='ok'), dict(role='assistant', content='reply')])
        self.assertEqual(a.previous_error(v, 3, v.prior(3))['status'], 'unknown')

    def test_explicit_error_retains_missing_response_issue(self):
        v = view([dict(role='user', content='work'), call('a'),
                  dict(role='tool', tool_call_id='a', content='{"error":"failed"}'),
                  call('b'), dict(role='assistant', content='reply')])
        result = a.previous_error(v, 4, v.prior(4))
        self.assertEqual(result['status'], 'explicit_error')
        self.assertIn('missing_tool_response', result['issues'])

    def test_native_error_flag_survives_normalized_field_loss(self):
        class Native:
            raw = [dict(role='tool', tool_call_id='c', content='text only')]
            unit = {'family': 'native_pi'}

            def originals(self, index):
                return [dict(type='message', message=dict(role='toolResult', toolCallId='c',
                                                         isError=True, content='text only'))]
        self.assertEqual(a.raw_response(Native(), 0), (True, None))

    def test_strict_short_stop_reuses_missing_and_conflict_rules(self):
        feature = dict(category='user_turn_end', previous_turn_tool_calls=2, call_count_status='known',
                       termination_status='missing', termination_value=None)
        self.assertEqual(a.classify_respond(feature), ('short_turn', None))
        self.assertEqual(a.classify_respond(dict(feature, termination_status='conflict')),
                         ('unknown', 'termination_conflict'))
        self.assertEqual(a.classify_respond(dict(feature, termination_status='known', termination_value='length')),
                         ('unknown', 'abnormal_short_user_turn'))


if __name__ == '__main__':
    unittest.main()
