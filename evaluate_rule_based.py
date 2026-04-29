import argparse
import json
from pathlib import Path

import cv2

from digit_segmentation import read_display, thresholding_digit
from roi_detection import crop_from_original, find_big_displays, preprocess


def predict_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    resized, _, edged = preprocess(image)
    displays = find_big_displays(edged)
    rois = crop_from_original(image, resized, displays, pad=5)

    predictions = []
    for _, _, _, _, roi in rois:
        thresh = thresholding_digit(roi)
        text, _ = read_display(thresh)
        predictions.append(text)

    return predictions


def compare_strings(expected, predicted):
    max_len = max(len(expected), len(predicted))
    correct = 0

    for index in range(max_len):
        expected_char = expected[index] if index < len(expected) else None
        predicted_char = predicted[index] if index < len(predicted) else None
        if expected_char == predicted_char:
            correct += 1

    return correct, max_len


def evaluate_dataset(image_dir, ground_truth):
    image_dir = Path(image_dir)

    image_count = 0
    detection_success = 0
    roi_exact_matches = 0
    roi_total = 0
    char_correct = 0
    char_total = 0
    mismatches = []

    for image_name, expected_rois in sorted(ground_truth.items()):
        image_count += 1
        image_path = image_dir / image_name

        try:
            predicted_rois = predict_image(image_path)
        except FileNotFoundError as exc:
            mismatches.append({"image": image_name, "error": str(exc)})
            continue

        if len(predicted_rois) == len(expected_rois):
            detection_success += 1

        max_rois = max(len(expected_rois), len(predicted_rois))
        image_mismatch = {
            "image": image_name,
            "expected": expected_rois,
            "predicted": predicted_rois,
            "roi_errors": [],
        }

        for index in range(max_rois):
            expected = expected_rois[index] if index < len(expected_rois) else ""
            predicted = predicted_rois[index] if index < len(predicted_rois) else ""

            roi_total += 1
            if expected == predicted:
                roi_exact_matches += 1
            else:
                image_mismatch["roi_errors"].append(
                    {
                        "roi_index": index,
                        "expected": expected,
                        "predicted": predicted,
                    }
                )

            correct, total = compare_strings(expected, predicted)
            char_correct += correct
            char_total += total

        if image_mismatch["roi_errors"] or len(predicted_rois) != len(expected_rois):
            mismatches.append(image_mismatch)

    return {
        "image_count": image_count,
        "detection_success": detection_success,
        "roi_exact_matches": roi_exact_matches,
        "roi_total": roi_total,
        "char_correct": char_correct,
        "char_total": char_total,
        "mismatches": mismatches,
    }


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Ground truth file must be a JSON object mapping image names to ROI strings.")

    normalized = {}
    for image_name, value in data.items():
        if image_name.startswith("_"):
            continue

        if isinstance(value, str):
            normalized[image_name] = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            normalized[image_name] = value
        else:
            raise ValueError(
                f"Ground truth for {image_name} must be a string or list of strings."
            )

    return normalized


def print_report(results):
    image_count = results["image_count"]
    detection_success = results["detection_success"]
    roi_exact_matches = results["roi_exact_matches"]
    roi_total = results["roi_total"]
    char_correct = results["char_correct"]
    char_total = results["char_total"]

    detection_rate = (100.0 * detection_success / image_count) if image_count else 0.0
    roi_accuracy = (100.0 * roi_exact_matches / roi_total) if roi_total else 0.0
    char_accuracy = (100.0 * char_correct / char_total) if char_total else 0.0

    print(f"Images evaluated: {image_count}")
    print(f"ROI count matched: {detection_success}/{image_count} ({detection_rate:.2f}%)")
    print(f"ROI exact match: {roi_exact_matches}/{roi_total} ({roi_accuracy:.2f}%)")
    print(f"Character accuracy: {char_correct}/{char_total} ({char_accuracy:.2f}%)")

    if not results["mismatches"]:
        print("No mismatches.")
        return

    print("\nMismatches:")
    for mismatch in results["mismatches"]:
        if "error" in mismatch:
            print(f"- {mismatch['image']}: {mismatch['error']}")
            continue

        print(
            f"- {mismatch['image']}: expected={mismatch['expected']} predicted={mismatch['predicted']}"
        )
        for roi_error in mismatch["roi_errors"]:
            print(
                f"  roi_{roi_error['roi_index']}: "
                f"expected='{roi_error['expected']}' predicted='{roi_error['predicted']}'"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the current rule-based display reader against labeled images."
    )
    parser.add_argument(
        "--images",
        default="./img/raw",
        help="Directory containing evaluation images. Default: ./img/raw",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="JSON file mapping image names to expected ROI strings.",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    results = evaluate_dataset(args.images, ground_truth)
    print_report(results)


if __name__ == "__main__":
    main()
