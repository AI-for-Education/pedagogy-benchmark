"""Language-specific prompts and configurations for multilingual benchmarking."""

LANGUAGE_PROMPTS = {
    'luganda': {
        'intro': 'Bino wammanga bibuuzo bya kulabirako eby\'okulondamu (n\'ebyanulo).',
        'instruction': 'Anukula ekibuuzo kino ekituufu ng\'ogoberera enkola y\'emu:',
        'final': 'Anukula ekibuuzo ekituufu kyokka.\nWandiika enukuta yokka ey\'ekyanulo kyo.\nKoma ddala ku nukuta.',
        'display_name': 'Luganda',
        'slug': 'Luganda',
    },
    'luganda_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Luganda.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Luganda (English Prompt)',
        'slug': 'Luganda_ep',
    },
    'swahili': {
        'intro': 'Yafuatayo ni maswali ya mfano ya kuchagua (pamoja na majibu).',
        'instruction': 'Jibu swali halisi lifuatalo kwa kutumia muundo uleule wa majibu:',
        'final': 'Jibu tu swali halisi.\nToa tu herufi ya jibu lako.\nKoma haswa baada ya herufi.',
        'display_name': 'Swahili',
        'slug': 'Swahili',
    },
    'swahili_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Swahili.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Swahili (English Prompt)',
        'slug': 'Swahili_ep',
    },
    'hausa': {
        'intro': 'Ga misalan tambayoyi masu zaɓi (tare da amsoshinsu).',
        'instruction': 'Amsa tambaya ta ainihi mai zuwa ta amfani da tsarin amsa iri ɗaya:',
        'final': 'Amsa tambayar gaske kawai.\nBada harafin amsarka kawai.\nTsaya daidai bayan harafin.',
        'display_name': 'Hausa',
        'slug': 'Hausa',
    },
    'hausa_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Hausa.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Hausa (English Prompt)',
        'slug': 'Hausa_ep',
    },
    'yoruba': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Yoruba.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Yoruba',
        'slug': 'Yoruba',
    },
    'yoruba_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Yoruba.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Yoruba (English Prompt)',
        'slug': 'Yoruba_ep',
    },
    'nyankore': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Nyankore.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Nyankore',
        'slug': 'Nyankore',
    },
    'nyankore_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Nyankore.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Nyankore (English Prompt)',
        'slug': 'Nyankore_ep',
    },
    'english': {
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'English',
        'slug': 'English',
    }
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
