"""Compatibility shim.

The public Codex source tree only ships ``codex.data.connections``, but the
pre-pickled NeuronDB published on Google Cloud Storage was created with an
internal variant that imports from ``codex.data.connections_v2``. Without this
module the pickle fails to deserialize with::

    ModuleNotFoundError: No module named 'codex.data.connections_v2'

We re-export the public Connections class under the name the pickle expects.
If the pickled class layout matches the public one, unpickling succeeds. If
attributes diverge in the future, this is the place to bridge them.
"""

from codex.data.connections import *  # noqa: F401,F403
from codex.data.connections import Connections  # noqa: F401

# The internal variant named the class ``ConnectionsV2``; the public tree
# kept ``Connections``. Expose both names so the pickle can find what it wants.
ConnectionsV2 = Connections
