import unittest
from unittest import mock

from PIL import Image

import server


class MatteEdgeCleanupTests(unittest.TestCase):
    def test_auto_key_color_recognizes_magenta_border(self):
        image = Image.new("RGB", (12, 12), (249, 7, 247))
        image.paste((255, 255, 255), (3, 3, 9, 9))

        color, ratio = server.dominant_border_key_color(image)

        self.assertEqual(color, (249, 7, 247))
        self.assertGreater(ratio, 0.9)

    def test_magenta_edge_color_is_recovered_from_alpha_composite(self):
        image = Image.new("RGBA", (1, 1), (153, 50, 203, 128))

        cleaned = server.despill_alpha_edges(image, (255, 0, 255), 1.0)

        red, green, blue, alpha = cleaned.getpixel((0, 0))
        self.assertEqual(alpha, 128)
        self.assertLessEqual(abs(red - 52), 1)
        self.assertLessEqual(abs(green - 100), 1)
        self.assertLessEqual(abs(blue - 151), 1)

    def test_green_edge_color_is_recovered_from_alpha_composite(self):
        image = Image.new("RGBA", (1, 1), (40, 168, 100, 128))

        cleaned = server.despill_alpha_edges(image, (0, 255, 0), 1.0)

        red, green, blue, alpha = cleaned.getpixel((0, 0))
        self.assertEqual(alpha, 128)
        self.assertLessEqual(abs(red - 80), 1)
        self.assertLessEqual(abs(green - 81), 1)
        self.assertLessEqual(abs(blue - 199), 1)

    def test_decontamination_preserves_alpha_and_opaque_pixels(self):
        image = Image.new("RGBA", (2, 1))
        image.putdata([(30, 60, 90, 255), (200, 10, 200, 4)])

        cleaned = server.despill_alpha_edges(image, (255, 0, 255), 1.0)

        self.assertEqual(cleaned.getpixel((0, 0)), (30, 60, 90, 255))
        self.assertEqual(cleaned.getpixel((1, 0)), (200, 10, 200, 4))

    def test_edge_bleed_replaces_transparent_key_rgb_without_changing_alpha(self):
        image = Image.new("RGBA", (5, 5), (255, 0, 255, 0))
        image.putpixel((2, 2), (40, 120, 200, 255))

        cleaned = server.bleed_transparent_edges(image, pixels=1)

        self.assertEqual(cleaned.getpixel((2, 2)), (40, 120, 200, 255))
        self.assertEqual(cleaned.getpixel((2, 1)), (40, 120, 200, 0))
        self.assertEqual(cleaned.getpixel((0, 0)), (255, 0, 255, 0))

    def test_magenta_chroma_pipeline_outputs_clean_soft_edge(self):
        image = Image.new("RGBA", (3, 1), (255, 0, 255, 255))
        image.putpixel((1, 0), (153, 50, 203, 255))

        cleaned = server.chroma_key_frame(
            image,
            key_rgb=(255, 0, 255),
            threshold=0,
            softness=202,
            despill_strength=1.0,
            halo_pixels=0,
        )

        red, green, blue, alpha = cleaned.getpixel((1, 0))
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 255)
        self.assertLess(red, 100)
        self.assertGreater(green, 70)
        self.assertLess(blue, 190)

    def test_chroma_pipeline_reports_generic_edge_cleanup(self):
        image = Image.new("RGB", (3, 1), (255, 0, 255))
        image.putpixel((1, 0), (80, 120, 160))

        _frames, key_rgb, info = server.apply_matte_pipeline(
            raw_images=[image],
            chroma_enabled=True,
            matte_mode="chroma",
            key_mode="manual",
            manual_key_hex="#ff00ff",
            threshold=10,
            softness=100,
            despill_strength=1.5,
            halo_pixels=0,
            ai_model="",
            ai_device="cpu",
            ai_resolution="auto",
            luma_black=0,
            luma_white=255,
            luma_gamma=1.0,
            luma_strength=1.0,
            corridorkey_enabled=False,
            corridorkey_screen="auto",
        )

        self.assertEqual(key_rgb, (255, 0, 255))
        self.assertEqual(info["despill_strength"], 1.0)
        self.assertTrue(info["edge_decontamination"])
        self.assertEqual(info["edge_bleed_pixels"], 2)

    def test_birefnet_pipeline_decontaminates_magenta_soft_edge(self):
        image = Image.new("RGB", (5, 5), (255, 0, 255))
        image.putpixel((2, 2), (153, 50, 203))
        alpha = Image.new("L", image.size, 0)
        alpha.putpixel((2, 2), 128)
        ai_info = {
            "model_key": "birefnet-hr-matting",
            "model_label": "BiRefNet HR-matting",
            "repo_id": "test/model",
            "device": "cuda",
            "resolution": 1024,
        }

        with mock.patch.object(
            server,
            "birefnet_alpha_mask",
            return_value=(alpha, ai_info),
        ):
            frames, key_rgb, info = server.apply_matte_pipeline(
                raw_images=[image],
                chroma_enabled=False,
                matte_mode="birefnet",
                key_mode="auto",
                manual_key_hex="#ff00ff",
                threshold=10,
                softness=100,
                despill_strength=1.0,
                halo_pixels=0,
                ai_model="birefnet-hr-matting",
                ai_device="cuda",
                ai_resolution="auto",
                luma_black=0,
                luma_white=255,
                luma_gamma=1.0,
                luma_strength=1.0,
                corridorkey_enabled=False,
                corridorkey_screen="auto",
            )

        red, green, blue, output_alpha = frames[0].getpixel((2, 2))
        self.assertEqual(key_rgb, (255, 0, 255))
        self.assertEqual(output_alpha, 128)
        self.assertLessEqual(abs(red - 52), 1)
        self.assertLessEqual(abs(green - 100), 1)
        self.assertLessEqual(abs(blue - 151), 1)
        self.assertTrue(info["edge_decontamination"])


if __name__ == "__main__":
    unittest.main()
