import sys
import signal

from analysis import analyze_sentinel1, analyze_sentinel2
from captioning import generate_captions
from datasets import choose_pair
from llm import compare_captions, create_answer, parse_intent
from multispectral import prepare_pair as prepare_ms_pair
from router import make_plan
from utils import ROOT, RoverError, geocode, save_json
from visualization import create_visuals


question = " ".join(sys.argv[1:]).strip()
if not question:
    question = input("Enter an Earth-observation question: ").strip()

if not question:
    print("No question was provided.")
    raise SystemExit(1)

try:
    intent = parse_intent(question)
    save_json("build/intent.json", intent)
    for filename in ("captions.json", "caption_comparison.json"):
        (ROOT / "build" / filename).unlink(missing_ok=True)

    print(f"Finding {intent['location']}...")
    bbox = geocode(intent["location"])

    plan = make_plan(intent, bbox)
    save_json("build/plan.json", plan)

    for folder_name, extensions in [
        ("rasters", {".tif", ".tiff"}),
        ("images", {".png", ".jpg", ".jpeg"}),
    ]:
        folder = ROOT / "build" / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for old_file in folder.iterdir():
            if old_file.is_file() and old_file.suffix.lower() in extensions:
                old_file.unlink()

    evidence = {
        "question": question,
        "event": intent["event"],
        "location": {"name": intent["location"], "bbox": bbox},
        "datasets_used": [],
        "scenes": {},
        "measurements": {},
        "quality": {},
        "rasters": {},
        "visuals": {},
        "earthdial_inputs": [],
        "captions": None,
        "multispectral": None,
        "caption_comparison": None,
        "warnings": [],
    }
    multispectral = None
    ms_unavailable = None

    for dataset in plan["datasets"]:
        if (intent["event"] == "water" and dataset == "sentinel-1"
                and "sentinel-2" in evidence["datasets_used"]):
            valid = evidence["quality"]["sentinel-2"]["valid_percent"]
            if valid is not None and valid >= 50:
                continue
        print(f"Searching {dataset}...")
        try:
            pair = choose_pair(dataset, plan)
            evidence["scenes"][dataset] = {
                "before": {
                    "id": pair["before"]["id"],
                    "datetime": pair["before"]["datetime"],
                    "overlap": pair["before"]["overlap"],
                    "cloud": pair["before"]["cloud"],
                },
                "after": {
                    "id": pair["after"]["id"],
                    "datetime": pair["after"]["datetime"],
                    "overlap": pair["after"]["overlap"],
                    "cloud": pair["after"]["cloud"],
                },
            }

            if dataset == "sentinel-1":
                result = analyze_sentinel1(pair, bbox, intent["event"])
            else:
                result = analyze_sentinel2(pair, bbox, intent["event"])
                def ms_deadline(_signum, _frame):
                    raise TimeoutError("Nine-band preparation exceeded 180 seconds")

                previous_alarm = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, ms_deadline)
                signal.alarm(180)
                try:
                    multispectral = prepare_ms_pair(pair, bbox, result["rasters"])
                    evidence["multispectral"] = multispectral
                    for role in ("before", "after"):
                        result["rasters"][f"s2_{role}_ms9"] = multispectral["images"][role]["path"]
                except (RoverError, TimeoutError) as error:
                    ms_unavailable = str(error)
                    evidence["warnings"].append(f"Multispectral captioning unavailable: {error}")
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_alarm)

            evidence["datasets_used"].append(dataset)
            evidence["measurements"].update(result["measurements"])
            evidence["quality"][dataset] = result["quality"]
            evidence["rasters"].update(result["rasters"])
            evidence["earthdial_inputs"].extend(result["earthdial_inputs"])
            evidence["warnings"].extend(pair["warnings"])
            evidence["warnings"].extend(result["warnings"])
        except (RoverError, OSError, ValueError) as error:
            evidence["warnings"].append(f"{dataset}: {error}")

    if not evidence["datasets_used"]:
        save_json("build/evidence.json", evidence)
        raise RoverError("No satellite dataset could be analyzed")

    evidence["visuals"] = create_visuals(
        evidence["rasters"], intent["location"], intent["event"]
    )

    if {"s2_before_rgb", "s2_after_rgb"}.issubset(evidence["rasters"]):
        print("Generating separate before and after captions...")
        try:
            evidence["captions"] = generate_captions(
                evidence["rasters"], evidence["scenes"].get("sentinel-2", {}),
                multispectral=multispectral, unavailable_reason=ms_unavailable,
            )
            if evidence["captions"].get("fallback_reason"):
                evidence["warnings"].append(
                    "RGB captions used: " + evidence["captions"]["fallback_reason"]
                )
            print("Comparing the saved captions...")
            evidence["caption_comparison"] = compare_captions(
                ROOT / "build" / "captions.json", question,
            )
            save_json("build/caption_comparison.json", evidence["caption_comparison"])
        except RoverError as error:
            evidence["warnings"].append(f"Caption comparison: {error}")
    else:
        evidence["warnings"].append(
            "Caption comparison: no Sentinel-2 before/after RGB pair was available"
        )

    save_json("build/evidence.json", evidence)

    answer = create_answer(question, evidence)
    if evidence["caption_comparison"] and evidence["caption_comparison"].get("reasoning"):
        answer += "\n\nCaption-based visual reasoning: " + (
            evidence["caption_comparison"]["reasoning"].strip()
        )
    if evidence["captions"] and evidence["captions"].get("mode") == "rgb_fallback":
        answer += "\nMultispectral captioning was unavailable; RGB captions were used."
    if evidence["caption_comparison"] is None:
        answer += "\nVisual caption comparison was unavailable; this summary uses raster measurements."
    if evidence["visuals"]:
        answer += "\n\nImages:\n"
        for label, filename in evidence["visuals"].items():
            answer += f"- {label.replace('_', ' ').title()}: {filename}\n"

    output = {
        "question": question,
        "intent": intent,
        "plan": plan,
        "evidence": evidence,
        "earthdial_output": None,
        "captions_file": "build/captions.json" if evidence["captions"] else None,
        "caption_comparison_file": (
            "build/caption_comparison.json" if evidence["caption_comparison"] else None
        ),
        "answer": answer.strip(),
    }
    save_json("build/output.json", output)
    print("\n" + answer.strip())

except RoverError as error:
    print(f"ROVER error: {error}", file=sys.stderr)
    raise SystemExit(1)
