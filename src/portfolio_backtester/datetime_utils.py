from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def get_bday_offset(holidays=None):
    """
    Returns a CustomBusinessDay offset that accounts for holidays.
    """
    if holidays is None:
        holidays = USFederalHolidayCalendar().holidays().to_pydatetime()
    return CustomBusinessDay(holidays=list(holidays))
