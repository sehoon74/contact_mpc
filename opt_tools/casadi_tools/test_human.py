from dataclasses import dataclass

class Hum:
    @dataclass
    class Input:
        tau: float = 0.

    @dataclass
    class State:
        test: Hum.Input = Hum.Input()

            
hum = Hum()
print(hum.State())
