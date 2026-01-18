# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

from typing import List

ALL_RELEVANT_TRAITS: List[str] = [
    "Smile",
    "Laughter",
    "Scowl",
    "Frowning eyebrows",
    "Raising eyebrows",
    "Frequent blinking",
    "Exaggerated eye opening",
    "Exaggerated eye closing",
    "Exaggerated mouth opening",
    "Mouth closed",
    "Lip corners up",
    "Lip corners down",
    "Lips protruded",
    "Lips retracted",
    "Chin moving upwards",
    "Chin moving downwards",
    "Gaze towards interlocutor",
    "Gaze up",
    "Gaze down",
    "Gaze sideways",
    "Nodding",
    "Head moving forward",
    "Head moving backward",
    "Head tilting",
    "Head side-turn",
    "Head shaking",
    "Head up",
    "Hands rubbing",
    "Hands scratching",
    "Hands touching head/body/etc",
    "Hands covering face/mouth/etc",
    "Crossing fingers",
    "Palms up",
    "Arms open",
    "Arms folded",
    "Shrugs",
]

classification_template_part1 = 'Please read the 2 texts below. Each of them contains an assesment of whether or not a person is lying. Each one of them contains arguments for and against both deception and truth. At the same time they both lead to a specific, more likely conclusion. Read them and output the final conclusions only. Do it in the following, example format: "Text 1: truth, Text 2: deception". The output values should be aligned with these texts...\n\nText 1:\n'

classification_template_part2 = "\n\nText 2:\n"

cue_f1_template = f"Please read the text below. Look for behaviors that are mentioned in the text from the following list: {repr(ALL_RELEVANT_TRAITS)}. Output those using the same exact wording as in the list, one per line. Don't ouput anything else. \n\nText:\n"

so_template_part1 = "Read those 2 texts describing the behavior of the same person and how it can be interpreted as a cue to deception/truthfulness. Score the logical overlap between those texts, you should pay attention to both the cues themselves and how they are interpreted and reasoned about. The score should be lower if e.g one of the texts focuses just on one interpretation of a specific cue etc. The score should be anywhere between 0.0 and 1.0 (both inclusive). Output the score only, nothing else. \n\nText 1:\n"

so_template_part2 = "\n\nText 2:\n"  # make sure they are aligned with the thesis!
