import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import server


class MarketParserTests(unittest.TestCase):
    def test_interpret_market_event_returns_valid_for_supported_alias_and_date(self):
        event = server.interpret_market_event("Highest temperature in Londres on March 16?")

        self.assertEqual(event["question"], "Highest temperature in Londres on March 16?")
        self.assertEqual(event["city"], "Londres")
        self.assertEqual(event["city_key"], "london")
        self.assertEqual(event["date"], "March 16")
        self.assertEqual(event["parse_status"], "valid")
        self.assertEqual(event["rule_confidence"], "HIGH")
        self.assertIn("city_matched_by_alias", event["parse_notes"])
        self.assertIn("date_inferred_from_title_only", event["parse_notes"])

    def test_interpret_market_event_returns_partial_for_unknown_city(self):
        event = server.interpret_market_event("Highest temperature in Atlantis on March 16?")

        self.assertEqual(event["city"], "Atlantis")
        self.assertEqual(event["city_key"], "atlantis")
        self.assertEqual(event["parse_status"], "partial")
        self.assertEqual(event["rule_confidence"], "MEDIUM")
        self.assertIn("city_not_canonical", event["parse_notes"])
        self.assertIn("date_inferred_from_title_only", event["parse_notes"])

    def test_interpret_market_event_returns_partial_for_unsupported_date_format(self):
        event = server.interpret_market_event("Highest temperature in London on 2026-03-16?")

        self.assertEqual(event["city_key"], "london")
        self.assertEqual(event["date"], "2026-03-16")
        self.assertEqual(event["parse_status"], "partial")
        self.assertEqual(event["rule_confidence"], "MEDIUM")
        self.assertIn("city_matched_by_alias", event["parse_notes"])
        self.assertIn("date_inferred_from_title_only", event["parse_notes"])
        self.assertIn("date_not_supported_by_current_baseline", event["parse_notes"])

    def test_interpret_market_event_returns_unknown_when_city_and_date_are_missing(self):
        event = server.interpret_market_event("Weather market?")

        self.assertEqual(event["city"], "")
        self.assertEqual(event["city_key"], "")
        self.assertEqual(event["date"], "")
        self.assertEqual(event["parse_status"], "unknown")
        self.assertEqual(event["rule_confidence"], "LOW")
        self.assertIn("city_not_extracted", event["parse_notes"])
        self.assertIn("date_not_extracted", event["parse_notes"])

    def test_normalize_city_query_recognizes_aliases(self):
        self.assertEqual(server.normalize_city_query("London (Heathrow)"), "london")
        self.assertEqual(server.normalize_city_query("londres"), "london")
        self.assertEqual(server.normalize_city_query("Tokyo (Haneda)"), "tokyo")

    def test_parse_event_title_extracts_supported_month_day_format(self):
        parsed = server.parse_event_title("Highest temperature in London on March 16?")

        self.assertEqual(parsed["question"], "Highest temperature in London on March 16?")
        self.assertEqual(parsed["city"], "London")
        self.assertEqual(parsed["date_label"], "March 16")

    def test_filter_market_events_filters_by_text_query(self):
        events = [
            server.interpret_market_event("Highest temperature in London on March 16?"),
            server.interpret_market_event("Highest temperature in Tokyo on March 17?"),
        ]

        filtered = server.filter_market_events(events, query="tokyo")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["city_key"], "tokyo")

    def test_filter_market_events_filters_by_city_alias(self):
        events = [
            server.interpret_market_event("Highest temperature in London on March 16?"),
            server.interpret_market_event("Highest temperature in Tokyo on March 17?"),
        ]

        filtered = server.filter_market_events(events, city_query="londres")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["city_key"], "london")

    def test_filter_market_events_applies_limit(self):
        events = [
            server.interpret_market_event("Highest temperature in London on March 16?"),
            server.interpret_market_event("Highest temperature in Tokyo on March 17?"),
            server.interpret_market_event("Highest temperature in New York on March 18?"),
        ]

        filtered = server.filter_market_events(events, limit="2")

        self.assertEqual(len(filtered), 2)

    def test_filter_market_events_ignores_invalid_limit(self):
        events = [
            server.interpret_market_event("Highest temperature in London on March 16?"),
            server.interpret_market_event("Highest temperature in Tokyo on March 17?"),
        ]

        filtered = server.filter_market_events(events, limit="abc")

        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
