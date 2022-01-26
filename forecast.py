import numpy as np

import pmdarima
import sktime.forecasting.ets


def arima(body):
    input = np.array(body.get('input'))

    arima = pmdarima.AutoARIMA(**body.get('params'))

    model = arima.fit(y=input)

    preds = list(model.predict(n_periods=body.get('h')))

    return {"predictions": preds}


def ets(body):
    input = np.array(body.get('input'))

    model = sktime.forecasting.ets.AutoETS(**body.get('params'))

    fitted = model.fit(input)

    # List of forecast horizon from 1 to length for sktime API
    fh = list(range(1, (body.get('h') + 1)))  # pylint: disable=C0301, C0103

    preds = list(fitted.predict(fh=fh).flatten())

    return {"predictions": preds}
