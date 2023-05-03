# Import dependencies
library(astsa)
library(tidyverse)
library(tswge)
library(tseries)
library(forecast)
library(knitr)
library(Metrics)

# Import data
# nfl = read.csv("nfl_attendance.csv")
# save(nfl, file = "nfl_attendance.RData")
load("nfl_attendance.RData")
head(nfl)

# Format data
nfl.cumulative = nfl %>%
  group_by(year, week) %>%
  summarize(attendance = sum(weekly_attendance, na.rm = TRUE)) %>%
  ungroup()
head(nfl.cumulative)

# Plot time series
nfl.cumulative.ts = ts(nfl.cumulative$attendance, start = c(2000, 1), frequency = 17)
tsplot(
  nfl.cumulative.ts, 
  main = "Figure 1: NFL Total Attendance Over Time",
  ylab = "Total Attendance"
)

# ACF and PACF
par(mfrow = c(2, 1))
acf(
  nfl.cumulative.ts,
  main = "Figure 2: NFL Total Attendance ACF"
)
pacf(
  nfl.cumulative.ts,
  main = "Figure 3: NFL Total Attendance PACF"
)
par(mfrow = c(1, 1))

# Test for stationarity
adf.test(nfl.cumulative.ts)

# Training/testing data split
nfl.training = nfl.cumulative %>%
  dplyr::filter(year < 2019)
nfl.training.ts = ts(nfl.training$attendance, start = c(2000, 1), frequency = 17)

nfl.testing = nfl.cumulative %>%
  dplyr::filter(year == 2019)
nfl.testing.ts = ts(nfl.testing$attendance, start = c(2019, 1), frequency = 17)

# Holt Winters Model

# Fit model
hwModel = HoltWinters(nfl.training.ts)
hwForecast = forecast(
  hwModel,
  h = 17
)

# Plot model
tsplot(
  nfl.cumulative.ts,
  main = "Figure 4: NFL Total Attendance Over Time Holt Winters Model",
  ylab = "Total Attendance"
)
lines(hwForecast$fitted, col = "red")
lines(hwForecast$mean, col = "blue")
legend(
  "topleft", 
  legend = c("Observed", "Fit", "Forecast"), 
  lty = 1, 
  col = c("black", "red", "blue")
)

# Evaluate model performance
hwResiduals = window(nfl.training.ts, start = 2001) - hwForecast$fitted
Box.test(hwResiduals)
rmse.hw = rmse(nfl.testing$attendance, hwForecast$mean)
rmse.hw

# SARIMA model

# Determine model order
auto.arima(nfl.training.ts, approximation = FALSE)

# Fit model
sarimaModel = arima(nfl.training.ts, order = c(1, 0, 1), seasonal = list(order = c(0, 1, 1), period = 17))
sarimaForecast = predict(sarimaModel, n.ahead = 17)

# Plot model
sarimaFit = nfl.training.ts - sarimaModel$residuals
tsplot(
  nfl.cumulative.ts,
  main = "Figure 5: NFL Total Attendance Over Time SARIMA Model",
  ylab = "Total Attendance"
)
lines(sarimaFit, col = "red")
lines(sarimaForecast$pred, col = "blue")
legend(
  "topleft", 
  legend = c("Observed", "Fit", "Forecast"), 
  lty = 1, 
  col = c("black", "red", "blue")
)

# Evalutate model performance
Box.test(sarimaModel$residuals)
rmse.sarima = rmse(nfl.testing$attendance, sarimaForecast$pred)
rmse.sarima

# Linear regression model

# Format data
nfl.cumulative2 = nfl.cumulative[18:nrow(nfl.cumulative),] %>%
  mutate(
    t = 18:nrow(nfl.cumulative),
    sin17 = sin((2 * pi * t) / 17),
    cos17 = cos((2 * pi * t) / 17),
    attendance.lag = nfl.cumulative$attendance[t - 17]
  )
head(nfl.cumulative2)

# Training/testing split
nfl.cumulative2.training = nfl.cumulative2 %>%
  filter(year < 2019)

nfl.cumulative2.testing = nfl.cumulative2 %>%
  filter(year == 2019)

# Fit model
lmModel = lm(attendance ~ sin17 + cos17 + attendance.lag, data = nfl.cumulative2.training)
summary(lmModel)

# Plot model
lmModel.predict = predict(lmModel, newdata = nfl.cumulative2.testing)
ts.lmModel = ts(lmModel$fitted.values, start = c(2001, 1), frequency = 17)
ts.lmPredict = ts(lmModel.predict, start = c(2019, 1), frequency = 17)
tsplot(
  nfl.cumulative.ts,
  main = "Figure 6: NFL Total Attendance Over Time Linear Regression Model",
  ylab = "Total Attendance"
)
lines(ts.lmModel, col = "red")
lines(ts.lmPredict, col = "blue")
legend(
  "topleft", 
  legend = c("Observed", "Fit", "Forecast"), 
  lty = 1, 
  col = c("black", "red", "blue")
)

# Evalutate model performance
Box.test(lmModel$residuals)
rmse.lm = rmse(nfl.cumulative2.testing$attendance, lmModel.predict)
rmse.lm

# Final results

# Plot all models
plot(
  nfl.testing.ts,
  main = "Figure 7: 2019 NFL Total Attendance vs Model Forecasts",
  ylab = "Total Attendance"
)
lines(hwForecast$mean, col = "red")
lines(sarimaForecast$pred, col = "blue")
lines(ts.lmPredict, col = "green")
legend(
  "bottomleft", 
  legend = c(
    "Observed", 
    "Holt Winters", 
    "SARIMA",
    "Linear Regression"
  ), 
  lty = 1, 
  col = c(
    "black", 
    "red", 
    "blue",
    "green"
  )
)

# Compare model RMSE's
data.frame(
  model = c(
    "Holt Winters",
    "SARIMA",
    "Linear Regression"
  ),
  rmse = c(
    rmse.hw,
    rmse.sarima,
    rmse.lm
  )
) %>%
  kable(
    col.names = c(
      "Model Type",
      "RMSE"
    ),
    caption = "RMSE of NFL Attendance Forecasts"
  )
