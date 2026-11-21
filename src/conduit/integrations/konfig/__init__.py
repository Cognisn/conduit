"""konfig settings loader and secret resolver, behind the konfig extra."""

# load_settings builds ConduitSettings from a konfig configuration's conduit section;
# KonfigSecretResolver resolves secret:// references through konfig's secrets store.
# Both take konfig's own Settings and Secrets objects, or anything with the same methods.

from conduit.integrations.konfig.loader import (
    DEFAULT_SECTION,
    KonfigSettingsSource,
    load_settings,
)
from conduit.integrations.konfig.resolver import (
    KONFIG_PLACEHOLDER,
    KonfigSecretResolver,
    KonfigSecretsSource,
)

__all__ = [
    "DEFAULT_SECTION",
    "KONFIG_PLACEHOLDER",
    "KonfigSecretResolver",
    "KonfigSecretsSource",
    "KonfigSettingsSource",
    "load_settings",
]
