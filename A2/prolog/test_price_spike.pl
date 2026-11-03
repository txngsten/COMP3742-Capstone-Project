/**
Student Name: Shivansh Pant
Student FAN: PANT0108
File: test_price_spike.pl
Date: 02-11-2026
Description: PlUnit tests for the price-spike recommendation rules and safety constraints.
Usage: swipl -g run_tests -t halt prolog/test_price_spike.pl
*/

:- use_module(library(plunit)).
:- prolog_load_context(directory, Dir),
   atom_concat(Dir, '/price_spike.pl', KB),
   use_module(KB).

%! case(+Prediction, +Probability, +GridDemand, +Battery, +Quality, -Result) is det.
%  Evaluates one case with a 0.25 decision threshold and no quality issues.
case(Prediction, Probability, Demand, Battery, Quality, Result) :-
    case(Prediction, Probability, Demand, Battery, Quality, [], Result).

case(Prediction, Probability, Demand, Battery, Quality, Issues0, R) :-
    assess(Prediction, Probability, 0.25, Demand, Battery, Quality, Issues0,
           Primary, Actions, Rules, _, Deferred, Violations, Confidence, Q, _),
    R = _{primary: Primary, actions: Actions, rules: Rules, deferred: Deferred,
          violations: Violations, confidence: Confidence, quality: Q}.

:- begin_tests(recommendations).

test(r1_defers_only_deferrable_loads) :-
    case(high_price, 0.9, normal, unavailable, valid, R),
    R.primary == defer_non_essential_loads,
    R.deferred \== [],
    forall(member(L, R.deferred), price_spike:site_load(L, deferrable)).

test(r1_and_r2_combine) :-
    case(high_price, 0.9, normal, available, valid, R),
    memberchk(defer_non_essential_loads, R.actions),
    memberchk(use_site_battery, R.actions).

test(r2_needs_site_battery) :-
    case(high_price, 0.9, normal, unavailable, valid, R),
    \+ memberchk(use_site_battery, R.actions).

test(r3_grid_stress_warns_and_still_defers) :-
    case(high_price, 0.9, critical, unavailable, valid, R),
    R.primary == issue_price_warning,
    memberchk(defer_non_essential_loads, R.actions),
    forall(price_spike:site_load(C, critical), \+ memberchk(C, R.deferred)).

test(r4_normal_operations) :-
    case(normal_price, 0.05, normal, unavailable, valid, R),
    R.actions == [maintain_operations],
    R.confidence == high.

test(r6_normal_price_critical_demand_is_covered) :-
    case(normal_price, 0.05, critical, unavailable, valid, R),
    R.primary == monitor_conditions.

test(r7_near_threshold_monitors) :-
    case(normal_price, 0.2, normal, unavailable, valid, R),
    memberchk(monitor_conditions, R.actions).

:- end_tests(recommendations).

:- begin_tests(safety).

test(r5_unavailable_prediction_requests_review) :-
    case(unavailable, 0.0, normal, unavailable, valid, R),
    R.primary == request_manual_review,
    R.actions == [request_manual_review, maintain_operations],
    R.rules == [r5_safety_default, r5_safety_default].

test(r5_invalid_data_blocks_economic_rules) :-
    case(high_price, 0.99, normal, available, invalid, R),
    R.primary == request_manual_review,
    \+ memberchk(defer_non_essential_loads, R.actions).

test(unknown_values_are_treated_as_invalid) :-
    case(spike, 3, normal, unavailable, valid, R),
    R.quality == invalid,
    R.primary == request_manual_review.

test(suspect_data_review_is_advisory) :-
    case(high_price, 0.9, normal, unavailable, valid, ["was_imputed"], R),
    R.quality == suspect,
    R.primary == defer_non_essential_loads,
    memberchk(request_manual_review, R.actions).

test(no_constraint_violations_in_any_valid_combination) :-
    forall(( member(P, [high_price, normal_price, unavailable]),
             member(D, [critical, normal]),
             member(B, [available, unavailable]),
             member(Q, [valid, suspect, invalid]),
             member(Pr, [0.05, 0.2, 0.5, 0.95]) ),
           ( case(P, Pr, D, B, Q, R), R.violations == [] )).

:- end_tests(safety).
