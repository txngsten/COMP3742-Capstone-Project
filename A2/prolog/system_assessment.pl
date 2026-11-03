/*
Student Names: Group P (integration by Hans Pujalte)
Student FAN: PUJA0009
File: system_assessment.pl
Date: 03-11-2026
Description: Coordinate subsystem forecasts into an advisory system assessment.
*/
:- module(system_assessment, [assess/9]).

% Priority is explicit: incomplete evidence must never imply normal conditions.
priority(incomplete, _, _, _, _, manual_review_required) :- !.
priority(_, critical, _, _, _, urgent_operator_review) :- !.
priority(_, high, _, _, _, operator_review_required) :- !.
priority(_, _, _, invalid, _, manual_review_required) :- !.
priority(_, _, high, _, _, operator_review_required) :- !.
priority(_, _, _, suspect, _, operator_review_required) :- !.
priority(_, medium, _, _, _, monitor_and_review) :- !.
priority(_, _, _, _, high_price, monitor_and_review) :- !.
priority(_, _, _, _, _, routine_monitoring).

% These actions coordinate advisory outputs, not physical control commands.
reason(incomplete, _, _, _, _, _, incomplete_evidence,
       'A subsystem failed, returned invalid evidence, or forecasts refer to different hours; resolve this before relying on the combined assessment.').
reason(_, critical, _, _, _, _, critical_network_stress,
       'Contact the system operator for urgent review of predicted critical network stress.').
reason(_, high, _, _, _, _, high_network_stress,
       'Review network conditions with the operator before acting on cost or renewable-share recommendations.').
reason(_, medium, _, _, _, _, medium_network_stress,
       'Review forecasts and monitor changing regional network conditions.').
reason(_, _, high, _, _, _, uncertain_demand,
       'Demand uncertainty is high; verify the forecast before relying on renewable coverage or scheduling decisions.').
reason(_, _, _, invalid, _, _, invalid_price_evidence,
       'Price assessment is invalid or unavailable; request manual review and maintain the safe operating schedule.').
reason(_, _, _, suspect, _, _, suspect_price_evidence,
       'Verify the flagged price inputs before applying operational recommendations.').
reason(_, _, _, _, high_price, _, price_warning,
       'A price spike is forecast; review the price-layer actions for deferrable loads, subject to operator approval.').
reason(_, _, _, _, _, insufficient, renewable_target_shortfall,
       'The renewable-share target is missed; review renewable availability and flexible demand. This is not evidence of total supply shortage or blackout risk.').
reason(_, _, _, _, _, sufficient, renewable_target_met,
       'The renewable-share target is met; total supply and reserve adequacy still require separate assessment.').

% Stateless arguments avoid interference between successive predictions or other KBs.
assess(Completeness, Network, Uncertainty, Quality, Price, Renewable, Priority, Rules, Texts) :-
    priority(Completeness, Network, Uncertainty, Quality, Price, Priority),
    findall(Id-Text, reason(Completeness, Network, Uncertainty, Quality, Price, Renewable, Id, Text), Pairs),
    pairs(Pairs, Rules, Texts).
pairs([], [], []).
pairs([Id-Text|Rest], [Id|Ids], [Text|Texts]) :- pairs(Rest, Ids, Texts).
