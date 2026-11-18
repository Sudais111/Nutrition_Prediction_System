import os
import subprocess
import sys
import unittest
import warnings

import app as nutrition_app


class NutritionAppTestCase(unittest.TestCase):
    def setUp(self):
        nutrition_app.app.config["TESTING"] = True
        self.client = nutrition_app.app.test_client()

    def test_home_page_loads_with_stats_and_rounded_chart_values(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("Nutrition Prediction System", html)
        self.assertIn("Total Records", html)
        self.assertIn("25", html)
        self.assertIn("Islamabad", html)
        self.assertIn("MICS", html)
        self.assertIn("NNS", html)
        # Ensure floating-point artifacts are rounded cleanly
        self.assertNotIn("20.674999999999997", html)
        self.assertNotIn("12.433333333333332", html)
        self.assertIn("20.67", html)
        self.assertIn("12.43", html)

    def test_get_predict_redirects_to_home(self):
        response = self.client.get("/predict", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")

    def test_all_measured_records_return_exact_values(self):
        for _, row in nutrition_app.df.iterrows():
            response = self.client.post(
                "/predict",
                data={
                    "geography": row["geography"],
                    "indicator": row["indicator"],
                    "survey": row["survey"],
                },
            )
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            expected_val = f"{round(float(row['value']), 2)}%"
            self.assertIn("Measured (Survey Record)", html)
            self.assertIn(expected_val, html)
            self.assertIn(f'value="{row["geography"]}" selected', html)
            self.assertIn(f'value="{row["indicator"]}" selected', html)
            self.assertIn(f'value="{row["survey"]}" selected', html)

    def test_all_unmeasured_combinations_use_ml_model_prediction(self):
        measured_keys = set(
            zip(
                nutrition_app.df["geography"],
                nutrition_app.df["indicator"],
                nutrition_app.df["survey"],
            )
        )
        unmeasured_count = 0
        for geo in nutrition_app.df["geography"].unique():
            for ind in nutrition_app.df["indicator"].unique():
                for srv in nutrition_app.df["survey"].unique():
                    if (geo, ind, srv) in measured_keys:
                        continue
                    unmeasured_count += 1
                    response = self.client.post(
                        "/predict",
                        data={"geography": geo, "indicator": ind, "survey": srv},
                    )
                    self.assertEqual(response.status_code, 200)
                    html = response.get_data(as_text=True)
                    self.assertIn("Modelled (ML Prediction)", html)
                    self.assertIn("Estimated range:", html)
                    self.assertIn(f'value="{geo}" selected', html)
                    self.assertIn(f'value="{ind}" selected', html)
                    self.assertIn(f'value="{srv}" selected', html)
        self.assertEqual(unmeasured_count, 95)

    def test_missing_form_fields_return_400_with_error(self):
        response = self.client.post("/predict", data={})
        self.assertEqual(response.status_code, 400)
        html = response.get_data(as_text=True)
        self.assertIn("Please select a geography, nutrition indicator, and survey.", html)

    def test_invalid_form_values_return_400_without_crashing(self):
        response = self.client.post(
            "/predict",
            data={
                "geography": "UnknownCity",
                "indicator": "Stunting",
                "survey": "NNS",
            },
        )
        self.assertEqual(response.status_code, 400)
        html = response.get_data(as_text=True)
        self.assertIn("Invalid selection provided", html)

    def test_app_imports_from_different_working_directory_without_warnings(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(project_dir)
        code = (
            "import sys, warnings; "
            f"sys.path.insert(0, {project_dir!r}); "
            "warnings.simplefilter('error'); "
            "import app; "
            "assert len(app.df) == 25"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=parent_dir,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Import from parent dir failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
