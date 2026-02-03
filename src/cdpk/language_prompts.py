"""Language-specific prompts and configurations for multilingual benchmarking."""

LANGUAGE_PROMPTS = {
    'luganda': {
        'intro': 'Bino wammanga bibuuzo bya kulabirako eby\'okulondamu (n\'ebyanulo).',
        'instruction': 'Anukula ekibuuzo kino ekituufu ng\'ogoberera enkola y\'emu:',
        'final': 'Anukula ekibuuzo ekituufu kyokka.\nWandiika enukuta yokka ey\'ekyanulo kyo.\nKoma ddala ku nukuta.',
        'display_name': 'Luganda',
        'slug': 'Luganda',
        'slug_new': 'Luganda_new'
    },
    'luganda_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Luganda.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Luganda (English Prompt)',
        'slug': 'Luganda_ep',
        'slug_new': 'Luganda_ep_new'
    },
    'swahili': {
        'intro': 'Yafuatayo ni maswali ya mfano ya kuchagua (pamoja na majibu).',
        'instruction': 'Jibu swali halisi lifuatalo kwa kutumia muundo uleule wa majibu:',
        'final': 'Jibu tu swali halisi.\nToa tu herufi ya jibu lako.\nKoma haswa baada ya herufi.',
        'display_name': 'Swahili',
        'slug': 'Swahili',
        'slug_new': 'Swahili_new'
    },
    'swahili_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Swahili.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Swahili (English Prompt)',
        'slug': 'Swahili_ep',
        'slug_new': 'Swahili_ep_new'
    },
    'hausa': {
        'intro': 'Ga misalan tambayoyi masu zaɓi (tare da amsoshinsu).',
        'instruction': 'Amsa tambaya ta ainihi mai zuwa ta amfani da tsarin amsa iri ɗaya:',
        'final': 'Amsa tambayar gaske kawai.\nBada harafin amsarka kawai.\nTsaya daidai bayan harafin.',
        'display_name': 'Hausa',
        'slug': 'Hausa',
        'slug_new': 'Hausa_new'
    },
    'hausa_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Hausa.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Hausa (English Prompt)',
        'slug': 'Hausa_ep',
        'slug_new': 'Hausa_ep_new'
    },
    'yoruba': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Yoruba.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Yoruba',
        'slug': 'Yoruba',
        'slug_new': 'Yoruba_new'
    },
    'yoruba_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Yoruba.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Yoruba (English Prompt)',
        'slug': 'Yoruba_ep',
        'slug_new': 'Yoruba_ep_new'
    },
    'nyankore': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Nyankore.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Nyankore',
        'slug': 'Nyankore',
        'slug_new': 'Nyankore_new'
    },
    'nyankore_ep': {
        'intro': 'The following instructions are in English, but the example questions and the final question to be answered are in Nyankore.',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'Nyankore (English Prompt)',
        'slug': 'Nyankore_ep',
        'slug_new': 'Nyankore_ep_new'
    },
    'english': {
        'intro': 'The following are example multiple choice questions (with answers).',
        'instruction': 'Answer the following real question using same answer format:',
        'final': 'Only answer the real question.\nOnly provide the letter for your answer.\nStop exactly after the letter.',
        'display_name': 'English',
        'slug': 'English',
        'slug_new': 'English_new'
    }
}

# Regex patterns for answer extraction (same for all languages)
REPAT = [
    r"^[^A-G]*([ABCDEFG])[^A-G]*$",
    r"^.*\b([ABCDEFG])\b.*$",
    r"^.*answer is ([ABCDEFG]).*$",
    r"^.*([ABCDEFG])\..*$"
]


def get_language_config(language):
    """Get configuration for a specific language.

    Args:
        language: Language code (e.g., 'luganda', 'swahili_ep')

    Returns:
        dict: Language configuration with keys: intro, instruction, final,
              display_name, slug, slug_new

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
