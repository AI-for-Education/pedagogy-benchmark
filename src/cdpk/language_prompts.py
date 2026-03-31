"""Language-specific prompts and configurations for multilingual benchmarking."""

LANGUAGE_PROMPTS = {
    'luganda': {
        'intro_ep': None,
        'intro': 'Bino wammanga bibuuzo bya kulabirako eby\'okulondamu (n\'ebyanulo).',
        'instruction': 'Anukula ekibuuzo kino ekituufu ng\'ogoberera enkola y\'emu:',
        'final': 'Anukula ekibuuzo ekituufu kyokka.\nWandiika enukuta yokka ey\'ekyanulo kyo.\nKoma ddala ku nukuta.',
        'display_name': 'Luganda',
        'slug': 'Luganda',
    },
    'luganda_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Luganda.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Luganda (English Prompt)',
        'slug': 'Luganda_ep',
    },
    'swahili': {
        'intro_ep': None,
        'intro': 'Yafuatayo ni maswali ya mfano ya kuchagua (pamoja na majibu).',
        'instruction': 'Jibu swali halisi lifuatalo kwa kutumia muundo uleule wa majibu:',
        'final': 'Jibu tu swali halisi.\nToa tu herufi ya jibu lako.\nKoma haswa baada ya herufi.',
        'display_name': 'Swahili',
        'slug': 'Swahili',
    },
    'swahili_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Swahili.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Swahili (English Prompt)',
        'slug': 'Swahili_ep',
    },
    'hausa': {
        'intro_ep': None,
        'intro': 'Ga misalan tambayoyi masu zaɓi (tare da amsoshinsu).',
        'instruction': 'Amsa tambaya ta ainihi mai zuwa ta amfani da tsarin amsa iri ɗaya:',
        'final': 'Amsa tambayar gaske kawai.\nBada harafin amsarka kawai.\nTsaya daidai bayan harafin.',
        'display_name': 'Hausa',
        'slug': 'Hausa',
    },
    'hausa_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Hausa.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Hausa (English Prompt)',
        'slug': 'Hausa_ep',
    },
    'yoruba': {
        'intro_ep': None,
        'intro': 'Eyi ni àwọn àpẹẹrẹ ìbéèrè olópọ̀ yíyàn (pẹ̀lú àwọn ìdáhùn).',
        'instruction': 'Dáhùn ìbéèrè gidi tí ó tẹ̀lé e ní lílo irú ọ̀nà ìdáhùn kan náà:',
        'final': 'Dáhùn ìbéèrè gidi nìkan.\nPèsè lẹ́tà fún ìdáhùn rẹ nìkan.\nDúró gẹ́lẹ́ lẹ́yìn lẹ́tà náà.',
        'display_name': 'Yoruba',
        'slug': 'Yoruba',
    },
    'yoruba_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Yoruba.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Yoruba (English Prompt)',
        'slug': 'Yoruba_ep',
    },
    'nyankore': {
        'intro_ep': None,
        'intro': 'Ebibuuzo ebi n\'eby\'okureeberaho by\'okutooranamo (n\'ebigaruzo).',
        'instruction': 'Garukamu ekibuuzo eki ekihikire orikukurata empandiika emwe:',
        'final': 'Garukamu ekibuuzo ekihikire kyonyini.\nOhandiika enyuguta yonka y\'eky\'okugarukamu kyawe.\nHemera ahanyuguta honka.',
        'display_name': 'Nyankore',
        'slug': 'Nyankore',
    },
    'nyankore_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Nyankore.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Nyankore (English Prompt)',
        'slug': 'Nyankore_ep',
    },
    'english': {
        'intro_ep': None,
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'English',
        'slug': 'English',
    },
    'dari': {
        'intro_ep': None,
        'intro': None,
        'instruction': None,
        'final': None,
        'display_name': 'Dari',
        'slug': 'Dari',
    },
    'dari_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Dari.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Dari (English Prompt)',
        'slug': 'Dari_ep',
    },
    'pashto': {
        'intro_ep': None,
        'intro': None,
        'instruction': None,
        'final': None,
        'display_name': 'Pashto',
        'slug': 'Pashto',
    },
    'pashto_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Pashto.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Pashto (English Prompt)',
        'slug': 'Pashto_ep',
    },
    'arabic': {
        'intro_ep': None,
        'intro': None,
        'instruction': None,
        'final': None,
        'display_name': 'Arabic',
        'slug': 'Arabic',
    },
    'arabic_ep': {
        'intro_ep': 'The following instructions are in English, but the example questions and the final question to be answered are in Arabic.',
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Arabic (English Prompt)',
        'slug': 'Arabic_ep',
    },
}

# Regex patterns for answer extraction (same for all languages)
REPAT = [
    r"^\s*([ABCDEFG])(?:[\.(?:\s*\n)]+.*)*$",
    r"^<think>[\s\S]*?</think>[\s\S]*?([ABCDEFG])$",  # for deepseek R1
    r"^## Step 1[\s\S]*?([ABCDEFG])[\.\s]*$",  # for llama 4
    r'[\s\S]*\n([A-G])"?$',  # for claude 4
]


def get_language_config(language):
    """Get configuration for a specific language.

    Args:
        language: Language code (e.g., 'luganda', 'swahili_ep')

    Returns:
        dict: Language configuration with keys: intro, instruction, final,
              display_name, slug

    Raises:
        ValueError: If language is not supported
    """
    if language not in LANGUAGE_PROMPTS:
        raise ValueError(
            f"Unknown language: {language}. "
            f"Available: {list(LANGUAGE_PROMPTS.keys())}"
        )
    return LANGUAGE_PROMPTS[language]


def list_available_languages():
    """List all available language codes.

    Returns:
        list: List of supported language codes
    """
    return list(LANGUAGE_PROMPTS.keys())
