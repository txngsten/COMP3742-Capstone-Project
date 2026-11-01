% Student Name: [Add your name]
% Student FAN: [Add your FAN]
% File: stress_rules.pl
% Date: 01-11-2026
% Description: Symbolic rules for network-stress interpretation and response.
% Usage: Consult from inference.py or SWI-Prolog.
% Licence: For COMP3742 assessment use.

:- dynamic predicted_risk/2.
:- dynamic prediction_confidence/2.
:- dynamic demand/2.
:- dynamic gross_demand/2.
:- dynamic imports/2.
:- dynamic exports/2.
:- dynamic import_share/2.
:- dynamic electricity_price/2.
:- dynamic curtailment/2.
:- dynamic renewable_percentage/2.
:- dynamic reserve_margin/2.
:- dynamic price_severe_threshold/1.
:- dynamic price_extreme_threshold/1.
:- dynamic curtailment_severe_threshold/1.
:- dynamic curtailment_extreme_threshold/1.
:- dynamic renewable_severe_threshold/1.

risk_rank(low, 0).
risk_rank(medium, 1).
risk_rank(high, 2).
risk_rank(critical, 3).

maximum_risk(A, B, Maximum) :-
    risk_rank(A, RankA),
    risk_rank(B, RankB),
    (RankA >= RankB -> Maximum = A ; Maximum = B).

% Reserve margin is optional because it is not present in the project dataset.
reserve_risk(T, critical) :- reserve_margin(T, Margin), Margin < 5, !.
reserve_risk(T, high) :- reserve_margin(T, Margin), Margin < 10, !.
reserve_risk(T, medium) :- reserve_margin(T, Margin), Margin < 15, !.
reserve_risk(_, low).

% Numeric severity thresholds are asserted from the saved model bundle so the
% symbolic layer remains aligned with the target definition used for training.
supply_risk(T, high) :-
    import_share(T, ImportShare),
    electricity_price(T, Price),
    price_extreme_threshold(PriceThreshold),
    ImportShare >= 0.20,
    Price >= PriceThreshold, !.
supply_risk(T, high) :-
    curtailment(T, Curtailment),
    curtailment_extreme_threshold(CurtailmentThreshold),
    Curtailment >= CurtailmentThreshold, !.
supply_risk(T, medium) :- import_share(T, ImportShare), ImportShare >= 0.15, !.
supply_risk(T, medium) :-
    curtailment(T, Curtailment),
    curtailment_severe_threshold(CurtailmentThreshold),
    Curtailment >= CurtailmentThreshold, !.
supply_risk(_, low).

adequacy_signal(T) :- import_share(T, Share), Share >= 0.15.
adequacy_signal(T) :-
    electricity_price(T, Price),
    price_severe_threshold(PriceThreshold),
    Price >= PriceThreshold.
congestion_signal(T) :-
    curtailment(T, Curtailment),
    curtailment_severe_threshold(CurtailmentThreshold),
    Curtailment >= CurtailmentThreshold.
congestion_signal(T) :-
    renewable_percentage(T, Percentage),
    renewable_severe_threshold(RenewableThreshold),
    Percentage >= RenewableThreshold,
    exports(T, Exports),
    Exports > 0.

stress_driver(T, mixed) :- adequacy_signal(T), congestion_signal(T), !.
stress_driver(T, adequacy) :- adequacy_signal(T), !.
stress_driver(T, congestion) :- congestion_signal(T), !.
stress_driver(T, general) :-
    \+ adequacy_signal(T),
    \+ congestion_signal(T).

final_risk(T, FinalRisk) :-
    predicted_risk(T, PredictedRisk),
    reserve_risk(T, ReserveRisk),
    supply_risk(T, SupplyRisk),
    maximum_risk(PredictedRisk, ReserveRisk, IntermediateRisk),
    maximum_risk(IntermediateRisk, SupplyRisk, FinalRisk).

operational_risk(T, Risk) :- final_risk(T, Risk).

reason(T, ml_prediction) :- predicted_risk(T, _).
reason(T, reserve_margin_below_5_percent) :-
    reserve_margin(T, Margin), Margin < 5.
reason(T, reserve_margin_below_10_percent) :-
    reserve_margin(T, Margin), Margin >= 5, Margin < 10.
reason(T, reserve_margin_below_15_percent) :-
    reserve_margin(T, Margin), Margin >= 10, Margin < 15.
reason(T, high_import_dependency_and_price) :-
    import_share(T, Share), Share >= 0.20,
    electricity_price(T, Price),
    price_extreme_threshold(PriceThreshold),
    Price >= PriceThreshold.
reason(T, severe_curtailment) :-
    curtailment(T, Curtailment),
    curtailment_extreme_threshold(CurtailmentThreshold),
    Curtailment >= CurtailmentThreshold.
reason(T, elevated_import_dependency) :-
    import_share(T, Share), Share >= 0.15, Share < 0.20.
reason(T, elevated_curtailment) :-
    curtailment(T, Curtailment),
    curtailment_severe_threshold(SevereThreshold),
    curtailment_extreme_threshold(ExtremeThreshold),
    Curtailment >= SevereThreshold,
    Curtailment < ExtremeThreshold.

interpretation(T, normal_operating_conditions) :- final_risk(T, low).
interpretation(T, elevated_conditions_require_attention) :- final_risk(T, medium).
interpretation(T, serious_conditions_require_intervention) :- final_risk(T, high).
interpretation(T, emergency_conditions_require_immediate_action) :-
    final_risk(T, critical).

recommendation(T, continue_normal_monitoring) :- final_risk(T, low).
recommendation(T, review_demand_and_generation_forecasts) :- final_risk(T, medium).
recommendation(T, verify_reserve_and_interconnector_availability) :-
    final_risk(T, medium).
recommendation(T, prepare_fast_start_generation) :-
    final_risk(T, high), stress_driver(T, Driver),
    member(Driver, [adequacy, mixed]).
recommendation(T, initiate_demand_response_readiness) :-
    final_risk(T, high), stress_driver(T, Driver),
    member(Driver, [adequacy, mixed]).
recommendation(T, investigate_binding_network_constraints) :-
    final_risk(T, high), stress_driver(T, Driver),
    member(Driver, [congestion, mixed]).
recommendation(T, consider_redispatch_or_storage_charging) :-
    final_risk(T, high), stress_driver(T, Driver),
    member(Driver, [congestion, mixed]).
recommendation(T, review_high_risk_conditions) :-
    final_risk(T, high), stress_driver(T, general).
recommendation(T, activate_emergency_reserves_subject_to_operator_approval) :-
    final_risk(T, critical).
recommendation(T, notify_authorised_system_operator) :- final_risk(T, critical).
recommendation(T, prepare_controlled_load_shedding_as_last_resort) :-
    final_risk(T, critical), stress_driver(T, Driver),
    member(Driver, [adequacy, mixed]).
