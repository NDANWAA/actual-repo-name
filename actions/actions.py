import unittest
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk import Tracker
from unittest.mock import Mock, patch

from actions import ActionGetBotStatus, ActionUpdateSymbol, ActionExplainConcept

class TestRasaActions(unittest.TestCase):
    def test_get_bot_status(self):
        action = ActionGetBotStatus()
        dispatcher = CollectingDispatcher()
        tracker = Tracker(sender_id="default", slots={}, latest_message={}, events=[], paused=False, followup_action=None, active_loop={}, latest_action_name=None)
        
        with patch('requests.get') as mocked_get:
            mocked_get.return_value.json.return_value = {"status": "running"}
            action.run(dispatcher, tracker, {})
            self.assertIn("The trading bot is currently running.", dispatcher.messages[0]['text'])

    def test_update_symbol(self):
        action = ActionUpdateSymbol()
        dispatcher = CollectingDispatcher()
        tracker = Tracker(sender_id="default", slots={"symbol": "AAPL"}, latest_message={}, events=[], paused=False, followup_action=None, active_loop={}, latest_action_name=None)
        
        with patch('requests.post') as mocked_post:
            action.run(dispatcher, tracker, {})
            self.assertIn("Trading symbol updated to AAPL.", dispatcher.messages[0]['text'])

    def test_explain_concept(self):
        action = ActionExplainConcept()
        dispatcher = CollectingDispatcher()
        tracker = Tracker(sender_id="default", slots={"concept": "Monte Carlo Simulation"}, latest_message={}, events=[], paused=False, followup_action=None, active_loop={}, latest_action_name=None)

        action.run(dispatcher, tracker, {})
        self.assertIn("A Monte Carlo simulation is a method", dispatcher.messages[0]['text'])

if __name__ == '__main__':
    unittest.main()
