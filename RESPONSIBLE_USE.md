# Responsible use

WiFi channel state information can reveal the presence, gestures and activities of
people, including through walls and without their awareness. Making sensing models more
robust to new rooms, which is what test-time adaptation aims at, lowers one barrier to
such use. We ask everyone who uses WiFi-TTA-Bench, its code or its derived artifacts to:

- sense only spaces whose occupants have given informed consent to being sensed;
- not use the benchmark or its methods for covert monitoring or surveillance;
- restrict access to sensing infrastructure and to the data it produces;
- report how often adaptation degrades a deployed model (negative-adaptation rate and
  source drop), not only its mean gain, before relying on it.

This is a statement of intent, not a licence term. The code and artifacts are released
under the licence in `LICENSE`, which this document does not restrict. The upstream
datasets (Widar3.0, NTU-Fi, SignFi) are governed by their own licences and terms.
