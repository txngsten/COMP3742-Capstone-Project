/**
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: uncertainty.pl
Date: 02-11-2026
Description: Rules and facts for explaining prediction with uncertainty.
*/

:- dynamic forecast/2.

% Band half-width as a fraction of the prediction
rel_width(R) :- forecast(pred, P), forecast(band, B), R is B / P.

% Uncertainty level
level(low) :- rel_width(R), R < 0.03.
level(medium) :- rel_width(R), R >= 0.03, R < 0.06.
level(high) :- rel_width(R), R >= 0.06.

% This hour historically had larger/smaller errors than the GP expected
reason(hard_hour) :- forecast(hour_scale, S), S > 1.2.
reason(easy_hour) :- forecast(hour_scale, S), S < 0.8.

% Demand moving fast right now
reason(steep_ramp) :- forecast(demand, D), forecast(lag_1h, L), abs(D - L) > 500.

% Forecast far from the same hour yesterday
reason(unlike_yesterday) :- forecast(pred, P), forecast(lag_23h, Y), abs(P - Y) / Y > 0.10.

% Current demand outside the training window's range
reason(out_of_range) :- forecast(demand, D), forecast(train_min, Lo), D < Lo.
reason(out_of_range) :- forecast(demand, D), forecast(train_max, Hi), D > Hi.

% Weekends are roughly 2/7 of the training window
reason(weekend) :- forecast(is_weekend, W), W > 0.5.

% Text outputs
text(hard_hour, 'this hour has historically had larger errors (daytime ramp / solar variability)').
text(easy_hour, 'this hour has historically been easy to predict (overnight, stable load)').
text(steep_ramp, 'demand is changing quickly right now, so the trend is harder to extrapolate').
text(unlike_yesterday, 'the forecast differs by more than 10% from the same hour yesterday').
text(out_of_range, 'current demand is outside the range seen in training, so the GP is extrapolating').
text(weekend, 'weekend hours are under-represented in the training window').

% Top-level query
explanation(Level, Texts) :-
    level(Level),
    findall(T, (reason(R), text(R, T)), Texts).