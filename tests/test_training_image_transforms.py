import unittest

from piper_train.training_image_transforms import append_image_transform_options


class TrainingImageTransformsTest(unittest.TestCase):
    def test_native_image_transform_config_is_forwarded_to_dataset_cli(self):
        command = ["lerobot-train"]
        training = {
            "image_transforms": {
                "enable": True,
                "max_num_transforms": 1,
                "random_order": False,
                "tfs": {
                    "ColorJitter": {
                        "type": "ColorJitter",
                        "kwargs": {
                            "brightness": 0.3,
                            "contrast": 0.3,
                            "saturation": 0.3,
                            "hue": 0.05,
                        },
                    }
                },
            }
        }

        append_image_transform_options(command, training)

        self.assertEqual(
            command,
            [
                "lerobot-train",
                "--dataset.image_transforms.enable=true",
                "--dataset.image_transforms.max_num_transforms=1",
                "--dataset.image_transforms.random_order=false",
                '--dataset.image_transforms.tfs={"ColorJitter":{"type":"ColorJitter",'
                '"kwargs":{"brightness":0.3,"contrast":0.3,"saturation":0.3,"hue":0.05}}}',
            ],
        )

    def test_missing_image_transform_config_adds_nothing(self):
        command = ["lerobot-train"]

        append_image_transform_options(command, {})

        self.assertEqual(command, ["lerobot-train"])


if __name__ == "__main__":
    unittest.main()
