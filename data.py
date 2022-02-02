import pandas as pd
import dateutil.easter as easter
import holidays as h

#======================================================
# EASTER
#======================================================

EASTER_DICT = {}
for year in range(2015,2020):
    EASTER_DICT[year] = easter.easter(year)
    
#======================================================
# SPECIAL DAYS
#======================================================

moms_day = {2015: pd.Timestamp(('2015-5-31')),
            2016: pd.Timestamp(('2016-5-29')),
            2017: pd.Timestamp(('2017-5-28')),
            2018: pd.Timestamp(('2018-5-27')),
            2019: pd.Timestamp(('2019-5-26'))}

wed_june = {2015: pd.Timestamp(('2015-06-24')),
            2016: pd.Timestamp(('2016-06-29')),
            2017: pd.Timestamp(('2017-06-28')),
            2018: pd.Timestamp(('2018-06-27')),
            2019: pd.Timestamp(('2019-06-26'))}

sun_nov = {2015: pd.Timestamp(('2015-11-1')),
        2016: pd.Timestamp(('2016-11-6')),
        2017: pd.Timestamp(('2017-11-5')),
        2018: pd.Timestamp(('2018-11-4')),
        2019: pd.Timestamp(('2019-11-3'))}

SPECIAL_DAYS_DICT = dict(moms_day = moms_day,
                    wed_june = wed_june,
                    sun_nov = sun_nov)

#======================================================
# COUNTRY HOLIDAYS
#======================================================

years = [2015, 2016, 2017, 2018, 2019]
finland = pd.DataFrame([dict(date = date, 
                            finland_holiday = event, 
                            country= 'Finland') for date, event in h.Finland(years=years).items()])

norway = pd.DataFrame([dict(date = date, 
                        norway_holiday = event, 
                        country= 'Norway') for date, event in h.Norway(years=years).items()])

sweden = pd.DataFrame([dict(date = date, 
                            sweden_holiday = event.replace(", Söndag", ""), 
                            country= 'Sweden') for date, event in h.Sweden(years=years).items() if event != 'Söndag'])
finland['date'] = finland['date'].astype("datetime64")
norway['date'] = norway['date'].astype("datetime64")
sweden['date'] = sweden['date'].astype("datetime64")

COUNTRY_HOLIDAYS_DICT = dict(finland = finland,
                            norway = norway,
                            sweden = sweden)

+56961559265