import http.cookiejar
import json
import os
import secrets
import sqlite3
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener


_TEST_DIRECTORY = tempfile.TemporaryDirectory(prefix="jobshield-auth-")
os.environ["JOBSHIELD_DB_PATH"] = os.path.join(_TEST_DIRECTORY.name, "integration.sqlite")
os.environ["JOBSHIELD_ADMIN_USERNAME"] = "test-admin-" + secrets.token_hex(4)
os.environ["JOBSHIELD_ADMIN_PASSWORD"] = secrets.token_urlsafe(24)

import app_server
import database


class AuthenticationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = app_server.HTTPServer(("127.0.0.1", 0), app_server.JobShieldRequestHandler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        _TEST_DIRECTORY.cleanup()

    def setUp(self):
        app_server.sessions.clear()
        database.init_db()
        conn = database.get_connection()
        with conn:
            conn.execute("DELETE FROM scan_history")
            conn.execute("DELETE FROM user_accounts")
        conn.close()
        self.user_jar = http.cookiejar.CookieJar()
        self.admin_jar = http.cookiejar.CookieJar()

    def test_existing_database_migration_preserves_accounts_and_unowned_scans(self):
        legacy_path = os.path.join(_TEST_DIRECTORY.name, "legacy.sqlite")
        conn = sqlite3.connect(legacy_path)
        conn.executescript(
            """
            CREATE TABLE user_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login_identifier TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_type TEXT NOT NULL,
                prediction INTEGER NOT NULL
            );
            INSERT INTO user_accounts
                (login_identifier, password_salt, password_hash, created_at)
            VALUES ('legacy@example.test', 'existing-salt', 'existing-hash', '2024-01-01');
            INSERT INTO scan_history (timestamp, source_type, prediction)
            VALUES ('2024-01-02', 'Direct Text', 1);
            """
        )
        conn.close()

        database.init_db(legacy_path)
        conn = sqlite3.connect(legacy_path)
        account = conn.execute(
            "SELECT login_identifier, username, email, role FROM user_accounts"
        ).fetchone()
        scan = conn.execute("SELECT id, user_id FROM scan_history").fetchone()
        conn.close()
        self.assertEqual(account[0], "legacy@example.test")
        self.assertEqual(account[1], "legacy-user-1")
        self.assertEqual(account[2], "legacy@example.test")
        self.assertEqual(account[3], "user")
        self.assertEqual(scan[0], 1)
        self.assertIsNone(scan[1])

    def request(self, path, method="GET", payload=None, cookies=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        cookie_jar = cookies if cookies is not None else http.cookiejar.CookieJar()
        opener = build_opener(HTTPCookieProcessor(cookie_jar))
        try:
            with opener.open(request, timeout=60) as response:
                body = response.read()
                return response.status, response.headers, body, response.geturl()
        except HTTPError as error:
            body = error.read()
            return error.code, error.headers, body, error.geturl()

    def json_request(self, path, method="GET", payload=None, cookies=None):
        status, headers, body, final_url = self.request(path, method, payload, cookies)
        return status, headers, json.loads(body or b"{}"), final_url

    def register(self, jar, username, email):
        return self.json_request(
            "/api/auth/register",
            "POST",
            {
                "full_name": "Integration Tester",
                "username": username,
                "email": email,
                "password": "TestPassword123!",
                "confirm_password": "TestPassword123!",
                "role": "admin",
            },
            jar,
        )

    def test_registration_login_profile_password_and_logout(self):
        status, _, result, _ = self.register(self.user_jar, "tester_one", "one@example.test")
        self.assertEqual(status, 201)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["redirect"], "/dashboard")
        status, _, session, _ = self.json_request("/api/auth/session", cookies=self.user_jar)
        self.assertEqual(status, 200)
        self.assertEqual(session["role"], "user")
        self.assertEqual(session["full_name"], "Integration Tester")

        duplicate_status, _, _, _ = self.register(
            http.cookiejar.CookieJar(), "TESTER_ONE", "other@example.test"
        )
        self.assertEqual(duplicate_status, 400)
        duplicate_email_status, _, _, _ = self.register(
            http.cookiejar.CookieJar(), "tester_two", "ONE@example.test"
        )
        self.assertEqual(duplicate_email_status, 400)

        profile_status, _, profile, _ = self.json_request(
            "/api/user/profile",
            "POST",
            {
                "full_name": "Updated Tester",
                "username": "tester_updated",
                "email": "updated@example.test",
            },
            self.user_jar,
        )
        self.assertEqual(profile_status, 200)
        self.assertEqual(profile["profile"]["full_name"], "Updated Tester")

        password_status, _, changed, _ = self.json_request(
            "/api/auth/change-password",
            "POST",
            {"current_password": "TestPassword123!", "new_password": "NewPassword123!"},
            self.user_jar,
        )
        self.assertEqual(password_status, 200)
        self.assertTrue(changed["changed"])

        self.request("/api/auth/logout", "POST", {}, self.user_jar)
        self.assertEqual(
            self.json_request("/api/user/profile", cookies=self.user_jar)[0],
            401,
        )
        login_status, _, login, _ = self.json_request(
            "/api/auth/login",
            "POST",
            {"identifier": "updated@example.test", "password": "NewPassword123!"},
            http.cookiejar.CookieJar(),
        )
        self.assertEqual(login_status, 200)
        self.assertEqual(login["redirect"], "/dashboard")

    def test_role_routes_and_admin_protected_apis(self):
        self.register(self.user_jar, "route_user", "route@example.test")
        user_page = self.request("/admin", cookies=self.user_jar)
        self.assertEqual(user_page[0], 200)
        self.assertTrue(user_page[3].endswith("/dashboard"))
        self.assertEqual(
            self.json_request("/api/admin/users", cookies=self.user_jar)[0],
            403,
        )
        self.assertEqual(self.request("/admin/users", cookies=self.user_jar)[3].endswith("/dashboard"), True)

        admin_status, _, admin_login, _ = self.json_request(
            "/api/auth/admin-login",
            "POST",
            {
                "identifier": os.environ["JOBSHIELD_ADMIN_USERNAME"],
                "password": os.environ["JOBSHIELD_ADMIN_PASSWORD"],
            },
            self.admin_jar,
        )
        self.assertEqual(admin_status, 200)
        self.assertEqual(admin_login["redirect"], "/admin")
        admin_page = self.request("/admin", cookies=self.admin_jar)
        self.assertEqual(admin_page[0], 200)
        self.assertTrue(admin_page[3].endswith("/admin"))
        users_status, _, users_result, _ = self.json_request(
            "/api/admin/users", cookies=self.admin_jar
        )
        self.assertEqual(users_status, 200)
        self.assertEqual(users_result["users"][0]["username"], "route_user")
        self.assertNotIn("password_hash", users_result["users"][0])

    def test_model_scans_are_owned_and_admin_sees_shared_records(self):
        self.register(self.user_jar, "owner_user", "owner@example.test")
        second_user_jar = http.cookiejar.CookieJar()
        self.register(second_user_jar, "other_user", "other@example.test")
        owner = database.authenticate_user("owner_user", "TestPassword123!")
        self.assertIsNotNone(owner)

        scan_status, _, scan, _ = self.json_request(
            "/api/analyze-text",
            "POST",
            {
                "text": "Urgent hiring. Pay a refundable registration fee by wire transfer to start immediately.",
                "title": "Remote Data Entry Clerk",
                "company": "Quick Hiring LLC",
                "source_type": "Direct Text",
                "file_name": "integration-scan.txt",
                "extraction_method": "Integration test input",
                "has_company_logo": False,
                "has_questions": False,
                "telecommuting": True,
            },
            self.user_jar,
        )
        self.assertEqual(scan_status, 200)
        self.assertIn("prediction", scan["analysis"])
        self.assertIn("branch_scores", scan["analysis"])
        scan_id = scan["scan_id"]

        owner_history = self.json_request("/api/history", cookies=self.user_jar)
        other_history = self.json_request("/api/history", cookies=second_user_jar)
        self.assertEqual(len(owner_history[2]["records"]), 1)
        self.assertEqual(other_history[2]["records"], [])
        self.assertEqual(
            self.request(f"/api/history/{scan_id}", "DELETE", cookies=second_user_jar)[0],
            404,
        )

        self.json_request(
            "/api/auth/admin-login",
            "POST",
            {
                "identifier": os.environ["JOBSHIELD_ADMIN_USERNAME"],
                "password": os.environ["JOBSHIELD_ADMIN_PASSWORD"],
            },
            self.admin_jar,
        )
        admin_history = self.json_request("/api/admin/detections", cookies=self.admin_jar)
        self.assertEqual(len(admin_history[2]["records"]), 1)
        admin_stats = self.json_request("/api/stats", cookies=self.admin_jar)
        self.assertEqual(admin_stats[2]["db_stats"]["total_scans"], 1)
        self.assertEqual(admin_stats[2]["total_users"], 2)
        user_stats = self.json_request("/api/stats", cookies=second_user_jar)
        self.assertEqual(user_stats[2]["db_stats"]["total_scans"], 0)
        self.assertNotIn("total_users", user_stats[2])

        suspend_status, _, _, _ = self.json_request(
            f"/api/admin/users/{owner['id']}/status",
            "POST",
            {"is_active": False},
            self.admin_jar,
        )
        self.assertEqual(suspend_status, 200)
        self.assertEqual(
            self.json_request("/api/user/profile", cookies=self.user_jar)[0],
            401,
        )

    def test_registration_confirmation_and_failed_admin_login(self):
        status, _, result, _ = self.json_request(
            "/api/auth/register",
            "POST",
            {
                "full_name": "Mismatch",
                "username": "mismatch_user",
                "email": "mismatch@example.test",
                "password": "TestPassword123!",
                "confirm_password": "DifferentPassword123!",
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("Passwords do not match", result["error"])

        status, _, _, _ = self.json_request(
            "/api/auth/admin-login",
            "POST",
            {"identifier": "not-admin", "password": "not-the-password"},
        )
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
