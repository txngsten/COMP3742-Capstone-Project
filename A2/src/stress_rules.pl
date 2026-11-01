% Student Name: [Add your name]
% Student FAN: [Add your FAN]
% File: stress_rules.pl
% Date: 01-11-2026
% Description: Symbolic rules for network-stress interpretation and response.
% Usage: Consult from inference.py or SWI-Prolog.
% Licence: For COMP3742 assessment use.

% stress_rules.pl
% Symbolic interpretation of predicted network stress.

:- dynamic predicted_risk/2.
:- dynamic import_share/2.
:- dynamic curtailment/2.

% Identify potential contributors to network stress
stress_driver(T, adequacy) :-
    import_share(T, Share),
    Share >= 0.15.

stress_driver(T, congestion) :-
    curtailment(T, Curtailment),
    Curtailment >= 250.

% Recommendations based on predicted risk
recommendation(T, continue_monitoring) :-
    predicted_risk(T, low).

recommendation(T, review_forecasts) :-
    predicted_risk(T, medium).

recommendation(T, investigate_supply_conditions) :-
    predicted_risk(T, high),
    stress_driver(T, adequacy).

recommendation(T, investigate_network_constraints) :-
    predicted_risk(T, high),
    stress_driver(T, congestion).

recommendation(T, review_high_risk_conditions) :-
    predicted_risk(T, high).

recommendation(T, notify_system_operator) :-
    predicted_risk(T, critical).
