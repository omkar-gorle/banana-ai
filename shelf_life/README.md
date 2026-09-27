# Future real shelf-life model

Your current classification dataset has no numeric `days_left` target.

For a real learned shelf-life model, collect longitudinal data.

Example CSV:

```csv
image_path,temperature_c,humidity_pct,days_left
data/longitudinal/b001_day0.jpg,27,65,5
data/longitudinal/b001_day1.jpg,27,66,4
data/longitudinal/b001_day2.jpg,28,68,3
data/longitudinal/b001_day3.jpg,28,70,2
data/longitudinal/b001_day4.jpg,29,72,1
data/longitudinal/b001_day5.jpg,29,75,0
```

The future model can combine:

- CNN visual embedding
- ripeness stage
- temperature
- humidity
- elapsed time
- banana variety, if available

with a regression or time-to-event model.

The current application deliberately does NOT fake this training target.
