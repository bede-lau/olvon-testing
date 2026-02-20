"""
Size recommendation engine using body measurements and garment dimensions.
Math-based approach: no ML needed.
"""

ELASTICITY_INDEX = {
    "cotton": 1.0,
    "spandex": 1.2,
    "polyester": 1.05,
    "cotton-poly blend": 1.03,
    "nylon": 1.1,
}

# Standard size ranges in centimeters
STANDARD_SIZES = {
    "S":   {"chest": (86, 91),   "waist": (71, 76),  "hip": (86, 91)},
    "M":   {"chest": (97, 102),  "waist": (81, 86),  "hip": (97, 102)},
    "L":   {"chest": (107, 112), "waist": (91, 97),  "hip": (107, 112)},
    "XL":  {"chest": (117, 122), "waist": (102, 107), "hip": (117, 122)},
    "XXL": {"chest": (127, 132), "waist": (112, 117), "hip": (127, 132)},
}


def compute_fit_score(user_measurement: float, garment_measurement: float, fabric: str = "cotton") -> float:
    """
    Compute how well a garment measurement fits a user measurement.
    Returns 0.0 (terrible fit) to 1.0 (perfect fit).
    Accounts for fabric elasticity.
    """
    elasticity = ELASTICITY_INDEX.get(fabric, 1.0)
    effective_garment = garment_measurement * elasticity

    diff = abs(user_measurement - effective_garment)
    tolerance = effective_garment * 0.1  # 10% tolerance band

    if diff <= tolerance:
        return 1.0 - (diff / tolerance) * 0.3  # 0.7-1.0 range for good fits
    else:
        overshoot = diff - tolerance
        return max(0.0, 0.7 - (overshoot / effective_garment) * 2.0)


def _estimate_from_height_weight(height_cm: float, weight_kg: float | None) -> dict:
    """
    Estimate chest/waist/hip from height and optional weight using empirical ratios.
    Returns estimated measurements in cm.
    """
    if weight_kg is not None and weight_kg > 0:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        # Empirical: chest ≈ height*0.52 + bmi*0.8, waist ≈ height*0.42 + bmi*1.2
        chest = height_cm * 0.52 + bmi * 0.8
        waist = height_cm * 0.42 + bmi * 1.2
        hip = height_cm * 0.52 + bmi * 0.6
    else:
        bmi = None
        # Height-only proportional estimates (assumes average BMI ~22)
        chest = height_cm * 0.52 + 22 * 0.8
        waist = height_cm * 0.42 + 22 * 1.2
        hip = height_cm * 0.52 + 22 * 0.6
    return {"chest": chest, "waist": waist, "hip": hip, "bmi": bmi}


def recommend_size(
    body_measurements: dict,
    garment_dimensions: dict | None = None,
    fabric: str = "cotton",
) -> dict:
    """
    Recommend a size based on body measurements.

    Args:
        body_measurements: dict with 'chest', 'waist', 'hip' in cm.
            Optional: 'height_cm' (float), 'weight_kg' (float).
            When height/weight provided, estimates are blended with mesh measurements.
        garment_dimensions: optional dict per size; if None, uses STANDARD_SIZES
        fabric: fabric type key from ELASTICITY_INDEX

    Returns:
        dict with recommended_size, confidence_score, fit_score, measurements
    """
    sizes = garment_dimensions or STANDARD_SIZES

    height_cm = body_measurements.get("height_cm")
    weight_kg = body_measurements.get("weight_kg")

    user_chest = body_measurements.get("chest", 96)
    user_waist = body_measurements.get("waist", 82)
    user_hip = body_measurements.get("hip", 96)

    bmi = None

    if height_cm is not None and height_cm > 0:
        hw_est = _estimate_from_height_weight(height_cm, weight_kg)
        bmi = hw_est["bmi"]

        if weight_kg is not None and weight_kg > 0:
            # Height + weight: 60% mesh / 40% estimate
            mesh_w, est_w = 0.6, 0.4
        else:
            # Height only: 70% mesh / 30% estimate
            mesh_w, est_w = 0.7, 0.3

        user_chest = user_chest * mesh_w + hw_est["chest"] * est_w
        user_waist = user_waist * mesh_w + hw_est["waist"] * est_w
        user_hip = user_hip * mesh_w + hw_est["hip"] * est_w

    best_size = None
    best_score = -1.0

    size_scores = {}
    for size_label, dims in sizes.items():
        chest_mid = sum(dims["chest"]) / 2
        waist_mid = sum(dims["waist"]) / 2
        hip_mid = sum(dims["hip"]) / 2

        chest_fit = compute_fit_score(user_chest, chest_mid, fabric)
        waist_fit = compute_fit_score(user_waist, waist_mid, fabric)
        hip_fit = compute_fit_score(user_hip, hip_mid, fabric)

        # Weighted average: chest matters most for upper-body garments
        combined = chest_fit * 0.5 + waist_fit * 0.3 + hip_fit * 0.2
        size_scores[size_label] = combined

        if combined > best_score:
            best_score = combined
            best_size = size_label

    # Confidence: how much better is best vs second-best
    sorted_scores = sorted(size_scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        confidence = min(1.0, 0.5 + (sorted_scores[0] - sorted_scores[1]) * 2.0)
    else:
        confidence = 0.5

    measurements = {
        "chest_cm": round(user_chest, 1),
        "waist_cm": round(user_waist, 1),
        "hip_cm": round(user_hip, 1),
    }
    if height_cm is not None:
        measurements["height_cm"] = round(height_cm, 1)
    if weight_kg is not None:
        measurements["weight_kg"] = round(weight_kg, 1)
    if bmi is not None:
        measurements["bmi"] = round(bmi, 1)

    return {
        "recommended_size": best_size,
        "confidence_score": round(confidence, 3),
        "fit_score": round(best_score, 3),
        "measurements": measurements,
        "all_scores": {k: round(v, 3) for k, v in size_scores.items()},
    }


if __name__ == "__main__":
    print("=== Sizing Logic Self-Test ===\n")

    # Test 1: Medium-sized person
    body = {"chest": 99, "waist": 83, "hip": 99}
    result = recommend_size(body, fabric="cotton")
    print(f"Body: {body}")
    print(f"Recommended: {result['recommended_size']} (confidence: {result['confidence_score']})")
    print(f"Fit score: {result['fit_score']}")
    print(f"All scores: {result['all_scores']}\n")
    assert result["recommended_size"] == "M", f"Expected M, got {result['recommended_size']}"

    # Test 2: Large person with stretchy fabric
    body2 = {"chest": 115, "waist": 100, "hip": 115}
    result2 = recommend_size(body2, fabric="spandex")
    print(f"Body: {body2} (spandex)")
    print(f"Recommended: {result2['recommended_size']} (confidence: {result2['confidence_score']})")
    print(f"All scores: {result2['all_scores']}\n")

    # Test 3: fit_score computation
    score = compute_fit_score(100, 100, "cotton")
    print(f"Perfect match fit score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"

    score2 = compute_fit_score(100, 80, "cotton")
    print(f"Poor match fit score: {score2}")
    assert score2 < 0.5, f"Expected < 0.5, got {score2}"

    # Test 4: Height + Weight
    body3 = {"chest": 99, "waist": 83, "hip": 99, "height_cm": 175, "weight_kg": 75}
    result3 = recommend_size(body3)
    print(f"\nBody with height/weight: {body3}")
    print(f"Recommended: {result3['recommended_size']}")
    print(f"BMI: {result3['measurements'].get('bmi')}")
    assert "bmi" in result3["measurements"], "BMI should be in measurements"
    assert "height_cm" in result3["measurements"], "height_cm should be in measurements"

    # Test 5: Height only (no weight)
    body4 = {"chest": 99, "waist": 83, "hip": 99, "height_cm": 180}
    result4 = recommend_size(body4)
    print(f"\nBody with height only: {body4}")
    print(f"Recommended: {result4['recommended_size']}")
    assert "height_cm" in result4["measurements"]
    assert "bmi" not in result4["measurements"]

    print("\n=== All self-tests passed ===")
