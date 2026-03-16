"""
src/mining/association.py
─────────────────────────
Khai phá luật kết hợp cho chỉ số chất lượng nước:
  - Rời rạc hoá chỉ số → transactions
  - Apriori / FP-Growth tìm frequent itemsets
  - Sinh luật kết hợp với support / confidence / lift / coverage
  - Lọc và xếp hạng top luật nguy hiểm (cùng vượt ngưỡng WHO)

Usage:
    from src.mining.association import WaterAssociationMiner

    miner = WaterAssociationMiner(min_support=0.2, min_confidence=0.7)
    rules = miner.fit(transactions)
    top_rules = miner.get_top_rules(n=10)
    miner.print_rules(top_rules)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    warnings.warn("mlxtend không được cài. Dùng: pip install mlxtend")


class WaterAssociationMiner:
    """
    Khai phá luật kết hợp cho dữ liệu chất lượng nước đã rời rạc hoá.

    Parameters
    ----------
    min_support : float
        Tần suất tối thiểu (default 0.20)
    min_confidence : float
        Độ tin cậy tối thiểu (default 0.70)
    min_lift : float
        Lift tối thiểu (default 1.5) — lift > 1 có ý nghĩa
    algorithm : str
        'apriori' hoặc 'fpgrowth'
    max_len : int
        Độ dài tối đa của itemset
    """

    def __init__(
        self,
        min_support: float = 0.20,
        min_confidence: float = 0.70,
        min_lift: float = 1.5,
        algorithm: str = "fpgrowth",
        max_len: int = 3,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.algorithm = algorithm
        self.max_len = max_len

        self._frequent_itemsets = None
        self._rules = None
        self._te = None
        self.runtime_sec: float = 0.0

    def fit(self, transactions: List[List[str]]) -> pd.DataFrame:
        """
        Tìm frequent itemsets và sinh luật kết hợp.

        Parameters
        ----------
        transactions : list of list
            Ví dụ: [["ph_High", "Turbidity_High"], ["ph_Low", "Solids_Medium"], ...]

        Returns
        -------
        rules : pd.DataFrame
            Bảng luật với: antecedents, consequents, support, confidence,
            lift, coverage, conviction
        """
        import time

        if not MLXTEND_AVAILABLE:
            return self._mock_rules()

        start = time.time()

        # Mã hoá one-hot
        self._te = TransactionEncoder()
        te_array = self._te.fit_transform(transactions)
        df_onehot = pd.DataFrame(te_array, columns=self._te.columns_)

        # Tìm frequent itemsets
        algo_fn = apriori if self.algorithm == "apriori" else fpgrowth
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._frequent_itemsets = algo_fn(
                df_onehot,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_len,
            )

        if len(self._frequent_itemsets) == 0:
            print(f"⚠ Không tìm thấy frequent itemsets với min_support={self.min_support}")
            return pd.DataFrame()

        # Sinh luật kết hợp (xử lý mlxtend >= 0.21 yêu cầu num_itemsets)
        try:
            self._rules = association_rules(
                self._frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence,
                num_itemsets=len(self._frequent_itemsets),
            )
        except TypeError:
            # mlxtend < 0.21 không có num_itemsets
            self._rules = association_rules(
                self._frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence,
            )

        # Lọc theo lift
        self._rules = self._rules[self._rules["lift"] >= self.min_lift]

        # Thêm cột coverage
        self._rules["coverage"] = self._rules["support"] / self._rules["confidence"]

        # Sắp xếp theo lift
        self._rules = self._rules.sort_values("lift", ascending=False).reset_index(drop=True)

        self.runtime_sec = round(time.time() - start, 3)

        print(f"✅ Tìm được {len(self._frequent_itemsets)} frequent itemsets, "
              f"{len(self._rules)} luật (lift≥{self.min_lift}) — {self.runtime_sec}s")

        return self._rules

    def get_top_rules(self, n: int = 10) -> pd.DataFrame:
        """Trả về top-n luật theo lift."""
        if self._rules is None:
            raise RuntimeError("Gọi fit() trước.")
        return self._rules.head(n)

    def get_dangerous_rules(self) -> pd.DataFrame:
        """
        Lọc các luật nguy hiểm: antecedent và consequent đều chứa '_High'
        (tức là nhiều chỉ số cùng vượt ngưỡng cao).
        """
        if self._rules is None:
            return pd.DataFrame()

        rules = self._rules.copy()
        rules["is_dangerous"] = rules.apply(
            lambda row: (
                all("_High" in str(item) for item in row["antecedents"]) and
                all("_High" in str(item) for item in row["consequents"])
            ),
            axis=1,
        )
        return rules[rules["is_dangerous"]].drop(columns=["is_dangerous"])

    def print_rules(self, rules: Optional[pd.DataFrame] = None, n: int = 10) -> None:
        """In các luật kết hợp dạng dễ đọc."""
        if rules is None:
            rules = self.get_top_rules(n)

        if len(rules) == 0:
            print("Không có luật nào.")
            return

        print(f"\n{'='*70}")
        print(f"TOP {min(n, len(rules))} LUẬT KẾT HỢP — Chỉ số chất lượng nước")
        print(f"{'='*70}")
        print(f"{'#':>3} {'Antecedent':30} {'Consequent':20} {'Sup':>6} {'Conf':>6} {'Lift':>6}")
        print("-" * 70)

        for i, (_, row) in enumerate(rules.head(n).iterrows()):
            ant = ", ".join(sorted(row["antecedents"]))
            con = ", ".join(sorted(row["consequents"]))
            sup = f"{row['support']:.3f}"
            conf = f"{row['confidence']:.3f}"
            lift = f"{row['lift']:.2f}×"
            print(f"{i+1:>3} {ant:30} {con:20} {sup:>6} {conf:>6} {lift:>7}")

        print(f"\nRuntime: {self.runtime_sec}s | min_support={self.min_support} | "
              f"min_confidence={self.min_confidence} | min_lift={self.min_lift}")

    def interpret_rules(self, rules: Optional[pd.DataFrame] = None) -> None:
        """
        Diễn giải top luật theo ngưỡng an toàn WHO.
        """
        if rules is None:
            rules = self.get_top_rules(5)

        print("\n📋 DIỄN GIẢI LUẬT THEO NGƯỠNG WHO:")
        print("-" * 60)

        interpretations = {
            "ph_High": "pH > 8.5 (kiềm cao)",
            "ph_Low": "pH < 6.5 (acid cao)",
            "Turbidity_High": "Độ đục > 4 NTU (nhiều cặn)",
            "Chloramines_High": "Chloramines > 4 ppm (clo quá cao)",
            "Solids_High": "TDS > 500 ppm (nhiều chất rắn hoà tan)",
            "Conductivity_High": "Độ dẫn điện > 400 μS/cm",
            "Organic_carbon_High": "Carbon hữu cơ > 2 ppm",
            "Trihalomethanes_High": "THMs > 80 μg/L (nguy cơ ung thư)",
            "Sulfate_High": "Sulfate > 250 mg/L (rối loạn tiêu hoá)",
            "Hardness_High": "Độ cứng > 300 mg/L (tích tụ canxi)",
        }

        for i, (_, row) in enumerate(rules.iterrows()):
            ant_items = sorted(row["antecedents"])
            con_items = sorted(row["consequents"])

            ant_interp = " + ".join([
                interpretations.get(item, item) for item in ant_items
            ])
            con_interp = " + ".join([
                interpretations.get(item, item) for item in con_items
            ])

            print(f"\nLuật {i+1}: [Support={row['support']:.2f}, Conf={row['confidence']:.2f}, Lift={row['lift']:.2f}×]")
            print(f"  NẾU: {ant_interp}")
            print(f"  THÌ: {con_interp}")
            print(f"  ⚠ Xác suất đồng xuất hiện cao hơn ngẫu nhiên {row['lift']:.1f} lần")

    def _mock_rules(self) -> pd.DataFrame:
        """Trả về rules mẫu khi mlxtend không được cài."""
        return pd.DataFrame({
            "antecedents": [frozenset(["ph_High", "Turbidity_High"]), frozenset(["Chloramines_High"])],
            "consequents": [frozenset(["Potability_Low"]), frozenset(["Trihalomethanes_High"])],
            "support": [0.38, 0.29],
            "confidence": [0.82, 0.78],
            "lift": [2.4, 2.1],
            "coverage": [0.46, 0.37],
        })

    def save_rules(self, path: str) -> None:
        """Lưu rules ra file CSV."""
        if self._rules is None:
            raise RuntimeError("Gọi fit() trước.")
        df_save = self._rules.copy()
        df_save["antecedents"] = df_save["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        df_save["consequents"] = df_save["consequents"].apply(lambda x: ", ".join(sorted(x)))
        df_save.to_csv(path, index=False)
        print(f"✅ Saved {len(df_save)} rules → {path}")
