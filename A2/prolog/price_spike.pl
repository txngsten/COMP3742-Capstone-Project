/**
Student Name: Shivansh Pant
Student FAN: PANT0108
File: price_spike.pl
Date: 02-11-2026
Description: Rules and facts that turn next-hour price-spike predictions into site actions.
*/

/** <module> Price-spike recommendation knowledge base

The machine learning subsystem (models/PriceModel.py) predicts whether the
next hour's NSW spot price will be high (> $300/MWh). This knowledge base
interprets that prediction for a large electricity user (a water utility
site) together with the grid's demand state and the site's own declared
assets, and returns recommended actions with the rules that produced them.

Grid facts and site facts are kept separate on purpose. grid_demand/1
describes stress on the NSW grid; it says nothing about how much water the
site must supply. site_battery/1 is a declared fact about the site; a high
renewable share on the grid does not give the site access to renewable
supply, so grid renewables are never used to justify an action.

Design principles:
  * Safety first: if the prediction is unavailable or the input data is
    invalid, only safe default actions are returned (rule r5).
  * Critical services are never deferred (hard restriction, checked by
    constraint_violation/1 on every result).
  * Every action is traceable to a named rule and a plain-English reason.

Called from Python through pyswip with assess/16.
*/

:- module(price_spike, [assess/16]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(pairs)).

% Facts about the case currently being evaluated (asserted per case).
:- dynamic prediction/1, probability/1, threshold/1, grid_demand/1,
           site_battery/1, data_quality/1, quality_issue/1,
           input_rejected/0.

% ---------------------------------------------------------------------------
% Domain vocabulary (facts)
% ---------------------------------------------------------------------------

%! price_condition(?Condition) is nondet.
%  Possible outputs of the ML subsystem. 'unavailable' means no trustworthy
%  prediction could be made.
price_condition(high_price).
price_condition(normal_price).
price_condition(unavailable).

%! grid_demand_condition(?Condition) is nondet.
%  NSW grid demand state: 'critical' at or above the training-set 95th
%  percentile. This is grid stress, not the site's own operational need.
grid_demand_condition(critical).
grid_demand_condition(normal).

%! battery_condition(?Condition) is nondet.
%  Whether the site's own battery can currently supply its running loads.
%  Declared by the site profile, not inferred from grid data.
battery_condition(available).
battery_condition(unavailable).

%! quality_condition(?Condition) is nondet.
%  Input-data state: suspect data contains imputed or flagged values.
quality_condition(valid).
quality_condition(suspect).
quality_condition(invalid).

%! recommendation_category(?Action) is nondet.
%  Every action the system may recommend.
recommendation_category(maintain_operations).
recommendation_category(defer_non_essential_loads).
recommendation_category(use_site_battery).
recommendation_category(issue_price_warning).
recommendation_category(request_manual_review).
recommendation_category(monitor_conditions).

%! action_priority(?Action, ?Priority) is nondet.
%  Lower numbers are more urgent; used to choose the primary action.
action_priority(request_manual_review, 1).
action_priority(issue_price_warning, 2).
action_priority(defer_non_essential_loads, 3).
action_priority(use_site_battery, 4).
action_priority(monitor_conditions, 5).
action_priority(maintain_operations, 6).

%! operational_restriction(?Restriction) is nondet.
%  Hard restrictions every recommendation must respect.
operational_restriction(critical_services_must_remain_active).
operational_restriction(non_essential_loads_can_be_delayed).

%! site_load(?Load, ?Class) is nondet.
%  Electrical loads at the example water utility site. Critical loads keep
%  drinking water safe and flowing; deferrable loads can shift by hours.
site_load(water_treatment_process_control, critical).
site_load(potable_water_supply_pumps, critical).
site_load(site_safety_and_scada_systems, critical).
site_load(reservoir_refill_pumps, deferrable).
site_load(sludge_dewatering, deferrable).
site_load(fleet_ev_charging, deferrable).

% ---------------------------------------------------------------------------
% Derived conditions
% ---------------------------------------------------------------------------

%! safety_override is semidet.
%  True when the ML output cannot be trusted, so economic rules must not run.
safety_override :- ( prediction(unavailable) ; data_quality(invalid) ), !.

%! confidence(-Level) is det.
%  'high' when the probability is at least halfway from the decision
%  threshold to certainty (in either direction), otherwise 'moderate'.
confidence(none) :- safety_override, !.
confidence(high) :-
    prediction(high_price), probability(P), threshold(T), P >= T + (1 - T) / 2, !.
confidence(high) :-
    prediction(normal_price), probability(P), threshold(T), P =< T / 2, !.
confidence(moderate).

%! near_threshold is semidet.
%  A 'normal' prediction whose probability is still close to the threshold.
near_threshold :-
    prediction(normal_price), probability(P), threshold(T), P > T / 2.

% ---------------------------------------------------------------------------
% Recommendation rules: rule(Id, Action, Explanation)
% ---------------------------------------------------------------------------

% R5 (safety, evaluated first): no trustworthy prediction -> human review
% and keep the site in its safe default state.
rule(r5_safety_default, request_manual_review,
     'Prediction unavailable or input data invalid; an operator must review before acting.') :-
    safety_override.
rule(r5_safety_default, maintain_operations,
     'Safe default: keep all loads running on the normal schedule until reviewed.') :-
    safety_override.

% R1: high price expected -> shift deferrable loads to a cheaper hour.
% Critical loads are untouched because only deferrable loads are listed.
rule(r1_defer_loads, defer_non_essential_loads,
     'High price expected next hour; deferrable loads can move to a cheaper hour. Critical services keep running.') :-
    \+ safety_override, prediction(high_price).

% R2: high price expected and the site battery is available -> use it.
rule(r2_use_site_battery, use_site_battery,
     'High price expected and the site battery is available; supply running loads from the battery instead of the grid.') :-
    \+ safety_override, prediction(high_price), site_battery(available).

% R3: high price during critical grid demand -> stronger warning. Grid stress
% makes reducing load more valuable, so R1 still applies; it is not a reason
% to keep deferrable loads running.
rule(r3_grid_stress_warning, issue_price_warning,
     'High price expected while grid demand is critical; grid stress makes a long or severe spike more likely, so warn the operator and confirm deferrals.') :-
    \+ safety_override, prediction(high_price), grid_demand(critical).

% R4: normal price and normal grid demand -> business as usual.
rule(r4_normal_operations, maintain_operations,
     'Normal price expected and grid demand is normal; continue the standard schedule.') :-
    \+ safety_override, prediction(normal_price), grid_demand(normal).

% R6: normal price predicted but the grid is stressed -> keep running, watch.
rule(r6_watch_grid_stress, monitor_conditions,
     'Normal price expected but grid demand is critical; grid stress can push prices up quickly, so keep watching.') :-
    \+ safety_override, prediction(normal_price), grid_demand(critical).
rule(r6_watch_grid_stress, maintain_operations,
     'No spike is predicted, so keep loads running on the normal schedule.') :-
    \+ safety_override, prediction(normal_price), grid_demand(critical).

% R7: 'normal' prediction close to the threshold -> flag a possible spike.
rule(r7_near_threshold, monitor_conditions,
     'Spike probability is close to the decision threshold; be ready to defer loads if prices rise.') :-
    \+ safety_override, near_threshold.

% R8: inputs contain imputed or flagged values -> act, but ask for a check.
rule(r8_suspect_data, request_manual_review,
     'Some inputs were imputed or flagged as suspect; verify them before relying on this recommendation.') :-
    \+ safety_override, data_quality(suspect).

% ---------------------------------------------------------------------------
% Constraints and result assembly
% ---------------------------------------------------------------------------

%! deferred_loads(-Loads) is det.
%  Loads to defer: only deferrable loads, and only if deferral was recommended.
deferred_loads(Loads) :-
    (   rule(_, defer_non_essential_loads, _)
    ->  findall(L, site_load(L, deferrable), Loads)
    ;   Loads = []
    ).

%! constraint_violation(-Violation) is nondet.
%  Succeeds for every way a result would break a hard restriction.
constraint_violation(critical_load_deferred(L)) :-
    deferred_loads(Loads), member(L, Loads), site_load(L, critical).
constraint_violation(no_action_recommended) :-
    \+ rule(_, _, _).

%! primary_action(+Actions, -Primary) is det.
%  Manual review leads only under a safety override; for suspect data it is
%  advisory, so the most urgent operational action leads instead.
primary_action(_, request_manual_review) :- safety_override, !.
primary_action(Actions, Primary) :-
    member(Primary, Actions), Primary \== request_manual_review, !.
primary_action(_, request_manual_review).

% ---------------------------------------------------------------------------
% Input validation and Python interface
% ---------------------------------------------------------------------------

%! load_case(+Prediction, +Probability, +Threshold, +Demand, +Battery,
%!           +Quality, +Issues) is det.
%  Replaces the asserted facts with a validated case. Any unexpected value
%  forces the case to invalid data quality, so it is handled by rule r5.
load_case(Prediction0, Probability0, Threshold0, Demand0, Battery0, Quality0, Issues) :-
    retractall(prediction(_)), retractall(probability(_)), retractall(threshold(_)),
    retractall(grid_demand(_)), retractall(site_battery(_)),
    retractall(data_quality(_)), retractall(quality_issue(_)),
    retractall(input_rejected),
    checked_atom(prediction, Prediction0, price_condition, unavailable, Prediction),
    checked_atom(grid_demand, Demand0, grid_demand_condition, normal, Demand),
    checked_atom(site_battery, Battery0, battery_condition, unavailable, Battery),
    checked_atom(data_quality, Quality0, quality_condition, invalid, Quality1),
    checked_number(probability, Probability0, 0.0, Probability),
    checked_number(threshold, Threshold0, 0.5, Threshold),
    (   is_list(Issues) -> forall(member(I, Issues), assertz(quality_issue(I))) ; true ),
    (   input_rejected -> Quality = invalid
    ;   quality_issue(_), Quality1 == valid -> Quality = suspect
    ;   Quality = Quality1
    ),
    assertz(prediction(Prediction)), assertz(probability(Probability)),
    assertz(threshold(Threshold)), assertz(grid_demand(Demand)),
    assertz(site_battery(Battery)), assertz(data_quality(Quality)).

%! checked_atom(+Name, +Raw, :Domain, +Fallback, -Value) is det.
%  Accepts Raw only if it is an atom in Domain. An unknown value is recorded
%  as a quality issue and the case is marked invalid.
checked_atom(Name, Raw, Domain, Fallback, Value) :-
    (   atom(Raw), call(Domain, Raw)
    ->  Value = Raw
    ;   Value = Fallback,
        format(string(Issue), "unrecognised ~w", [Name]),
        assertz(quality_issue(Issue)),
        assertz(input_rejected)
    ).

%! checked_number(+Name, +Raw, +Fallback, -Value) is det.
%  Accepts Raw only if it is a number in [0, 1]; anything else invalidates the case.
checked_number(Name, Raw, Fallback, Value) :-
    (   number(Raw), Raw >= 0, Raw =< 1
    ->  Value = Raw
    ;   Value = Fallback,
        format(string(Issue), "~w missing or outside 0-1", [Name]),
        assertz(quality_issue(Issue)),
        assertz(input_rejected)
    ).

%! assess(+Prediction, +Probability, +Threshold, +GridDemand, +SiteBattery,
%!        +Quality0, +Issues0, -Primary, -Actions, -RuleIds, -Explanations,
%!        -Deferred, -Violations, -Confidence, -Quality, -Issues) is det.
%  Evaluates one case and returns the recommendation as plain lists so that
%  pyswip can convert every output to Python values.
assess(Prediction, Probability, Threshold, Demand, Battery, Quality0, Issues0,
       Primary, Actions, RuleIds, Explanations, Deferred, Violations,
       Confidence, Quality, Issues) :-
    load_case(Prediction, Probability, Threshold, Demand, Battery, Quality0, Issues0),
    findall(Id-Text, rule(Id, _, Text), Fired),
    pairs_keys_values(Fired, RuleIds, Explanations),
    findall(A, rule(_, A, _), Raw),
    sort(Raw, Unique),
    map_list_to_pairs(action_priority, Unique, Keyed),
    keysort(Keyed, Sorted),
    pairs_values(Sorted, Actions),
    primary_action(Actions, Primary),
    deferred_loads(Deferred),
    findall(V, constraint_violation(V), Violations0),
    maplist(term_string, Violations0, Violations),
    confidence(Confidence),
    once(data_quality(Quality)),
    findall(I, quality_issue(I), Issues).
