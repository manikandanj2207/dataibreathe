from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from rasa_core.actions.action import Action; rasa-core 10 version;

import logging

from rasa_core_sdk import Action

from weather import Weather, Unit

logging.getLogger('rasa_core_sdk').setLevel(logging.ERROR);
logging.getLogger('urllib3').setLevel(logging.ERROR);

class ActionGetWeather(Action):
    def name(self):
        return 'action_get_weather'

    def run(self, dispatcher, tracker, domain):
        weather = Weather(unit=Unit.CELSIUS)
        gpe = ('Auckland', tracker.get_slot('GPE'))[bool(tracker.get_slot('GPE'))]
        result = weather.lookup_by_location(gpe)
        if result:
            condition = result.condition
            city = result.location.city
            country = result.location.country
            dispatcher.utter_message('It\'s ' + condition.text + ' and ' + condition.temp + ' C in ' + city + ', ' + country + '.');
        else:
            dispatcher.utter_message('We did not find any weather information for ' + gpe + '. Search by city name.')
        return
