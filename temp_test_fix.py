"""Test that other frequencies are not modified."""
        for freq in ["D", "W", "Q", "A", "Y"]:
            controller = self.create_time_based_timing({"rebalance_frequency": freq})
            expected_freq = freq
            if freq == "Q":
                expected_freq = "QE"
            elif freq == "W":
                expected_freq = "W-MON"
            elif freq == "A":
                expected_freq = "YE"
            assert controller.frequency == expected_freq