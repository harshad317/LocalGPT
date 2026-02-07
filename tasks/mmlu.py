"""
The MMLU dataset.
https://huggingface.co/datasets/cais/mmlu
"""

from datasets import load_dataset

try:
    from .common import Task, render_mc
except ImportError:
    import tasks.common as _tc

    Task = getattr(_tc, "Task", None) or getattr(_tc, "BaseTask", None) or getattr(_tc, "TaskBase", None)
    if Task is None:
        class Task:  # fallback minimal Task
            def __init__(self, start=0, stop=None, step=1):
                self.start = start
                self.stop = stop
                self.step = step

            @property
            def eval_type(self):
                raise NotImplementedError

            def num_examples(self):
                raise NotImplementedError

            def get_example(self, index):
                raise NotImplementedError

            def __len__(self):
                start = self.start
                stop = self.num_examples() if self.stop is None else self.stop
                step = self.step
                span = stop - start
                return (span + step - 1) // step

            def __getitem__(self, index: int):
                physical_index = self.start + index * self.step
                return self.get_example(physical_index)

            def evaluate(self, problem, completion):
                raise NotImplementedError

    render_mc = getattr(_tc, "render_mc", None)
    if render_mc is None:
        def render_mc(question, letters, choices):
            query = f"Multiple Choice question: {question}\n"
            query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
            query += "\nRespond only with the letter of the correct answer."
            return query

MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)

MMLU_SUBJECT_GROUPS = {
    # STEM/IT-focused subject groupings for targeted upweighting.
    "physics": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "high_school_physics",
    ],
    "biology": [
        "anatomy",
        "college_biology",
        "high_school_biology",
        "medical_genetics",
        "nutrition",
        "virology",
    ],
    "engineering": [
        "electrical_engineering",
    ],
    "cs": [
        "college_computer_science",
        "high_school_computer_science",
        "machine_learning",
    ],
    "it": [
        "computer_security",
    ],
}

_KNOWN_SUBJECTS = set(MMLU_SUBJECTS)


def _extract_subject(row):
    for key in ("subject", "subj", "topic", "category", "domain", "field", "task", "task_name", "subject_name"):
        val = row.get(key)
        if isinstance(val, str):
            val = val.strip()
            if val:
                return val
    for key in ("name", "group", "dataset"):
        val = row.get(key)
        if isinstance(val, str):
            val = val.strip()
            if val in _KNOWN_SUBJECTS:
                return val
    return ""

class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = MMLU_SUBJECTS

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train must be split into train"
        self.subset = subset
        self.split = split
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
        if subset == "auxiliary_train":
            # I don't understand why but the auxiliary_train rows have some weird additional 'train' wrapper.
            # Preserve subject metadata if it lives outside the wrapper.
            def _unwrap_aux(row):
                inner = row.get("train") or {}
                if not isinstance(inner, dict):
                    inner = {}
                if "subject" not in inner:
                    for key in ("subject", "subj", "topic", "category"):
                        if key in row and row[key] not in (None, ""):
                            inner = {**inner, "subject": row[key]}
                            break
                return inner
            self.ds = self.ds.map(_unwrap_aux, remove_columns=['train'])

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def _row_to_conversation(self, row):
        question = row["question"] # the question text
        choices = row["choices"] # the text of each choice
        answer = row["answer"] # index of the answer, e.g. 0,1,2,3 (for A,B,C,D)
        subject = _extract_subject(row) # e.g. "college_biology", "college_chemistry", etc.
        assert len(choices) == 4, "MMLU should have 4 choices"
        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject, # might be useful later for grouping metrics by subject
            "letters": self.letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def get_example(self, index):
        row = self.ds[index]
        return self._row_to_conversation(row)

    def evaluate(self, conversation, assistant_response):
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message


class MMLUSubjects(MMLU):
    def __init__(self, subjects, subset="auxiliary_train", split="train", **kwargs):
        self.subjects = list(subjects)
        super().__init__(subset=subset, split=split, **kwargs)
        subject_set = set(self.subjects)

        def _build_indices():
            return [i for i, row in enumerate(self.ds) if _extract_subject(row) in subject_set]

        self.indices = _build_indices()
        if not self.indices and subset == "auxiliary_train":
            try:
                self.ds = load_dataset("cais/mmlu", "all", split="auxiliary_train").shuffle(seed=42)
                self.subset = "all"
                self.split = "auxiliary_train"
                self.indices = _build_indices()
            except Exception:
                pass
        if not self.indices:
            raise ValueError(f"No MMLU rows found for subjects: {self.subjects}")

    def num_examples(self):
        return len(self.indices)

    def get_example(self, index):
        row = self.ds[self.indices[index]]
        return self._row_to_conversation(row)
