import odc

# A very basic test of the ODC model

def test_model(sample):
    curve = odc.ODC(pO2=sample['pO2'],
                    sO2=sample['sO2'],
                    pH=sample['pH'],
                    pCO2=sample['pCO2'],
                    FCOHb=0,
                    FMetHb=0,
                    FHbF=0)
    print("cDPG - Expected: {}, Calculated: {}".format(sample['expected_output']['cDPG'], curve.cDPG))
    print("p50 - Expected: {}, Calculated: {}".format(sample['expected_output']['p50'], curve.p50))

sample_1 = {
            "pH": 7.45,
            "pCO2": 5.4,
            "pO2": 12.2,
            "sO2": 0.97,
            "T": 37.0,
            "expected_output": {
                "p50": 3.51,
                "cDPG": 5.6
                }
            }

sample_2 = {
        "pH": 7.48,
        "pCO2": 4.7,
        "pO2": 6.0,
        "sO2": 0.81,
        "T": 37.0,
        "expected_output": {
                "p50": 3.55,
                "cDPG": 6.3
            }
        }

def main():
    test_model(sample_1)
    test_model(sample_2)

if __name__ == '__main__':
    main()
