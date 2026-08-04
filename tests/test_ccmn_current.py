import base64
import json
import unittest
from unittest.mock import Mock

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from domestic_prices.ccmn_current import AES_KEY, current_row, fetch_current_prices


def encrypt_payload(value):
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(json.dumps(value, ensure_ascii=False).encode()) + padder.finalize()
    encryptor = Cipher(algorithms.AES(AES_KEY), modes.ECB()).encryptor()
    return base64.b64encode(encryptor.update(padded) + encryptor.finalize()).decode()


class CcmnCurrentTests(unittest.TestCase):
    def test_fetch_current_prices_decrypts_public_payload(self):
        response = Mock()
        response.json.return_value = {
            "success": True,
            "body": {
                "showPriceList": encrypt_payload(
                    [[
                        {
                            "marketName": "长江现货",
                            "productSortName": "1#铜",
                            "avgPrice": 106940,
                            "publishDate": "2026-08-04",
                        },
                        {
                            "marketName": "长江现货",
                            "productSortName": "A00铝",
                            "avgPrice": 23720,
                            "publishDate": "2026-08-04",
                        },
                    ]]
                )
            },
        }
        session = Mock()
        session.get.return_value = response

        payload = fetch_current_prices(session=session)

        self.assertEqual(payload["date"], "2026-08-04")
        self.assertEqual(current_row(payload)["1#铜"], 106940)
        self.assertEqual(current_row(payload)["A00铝"], 23720)


if __name__ == "__main__":
    unittest.main()
