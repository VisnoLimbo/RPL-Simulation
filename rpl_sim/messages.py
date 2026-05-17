"""RPL control message definitions (RFC 6550).

Messages are modelled as plain dataclasses so they can be inspected,
logged, and serialised without heavyweight infrastructure.

Addressing assumption: Route-over with IPv6 prefixes.  Each node
carries a /128 address fd00::<node_id>.  The DODAG root advertises
the covering /64 so upward and downward routes are derivable from the
DODAG topology.  This follows the RPL route-over recommendation
(RFC 6550 §3, §9.9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ── ICMPv6 code points (RFC 6550 §6) ─────────────────────────────────────────
MSG_DIS = 0x00   # DODAG Information Solicitation
MSG_DIO = 0x01   # DODAG Information Object
MSG_DAO = 0x02   # Destination Advertisement Object
MSG_DAO_ACK = 0x03  # DAO Acknowledgment

# ── Rank constants ─────────────────────────────────────────────────────────────
INFINITE_RANK: int = 0xFFFF
MIN_HOP_RANK_INCREASE: int = 256   # DEFAULT_MIN_HOP_RANK_INCREASE (RFC 6550 §17)
ROOT_RANK: int = MIN_HOP_RANK_INCREASE  # Rank 1 × 256 for the DODAG root
DEFAULT_STEP_OF_RANK: int = 3          # OF0 default (RFC 6552 §4.1)


@dataclass
class RPLMessage:
    """Base class for all RPL control messages."""

    msg_type: int
    src_id: int
    timestamp: float = 0.0
    size_bytes: int = 0   # Wire-size proxy for overhead accounting


@dataclass
class DIS(RPLMessage):
    """DODAG Information Solicitation (RFC 6550 §6.3).

    Multicast by an unassociated node to solicit a DIO from neighbours.
    Optionally carries a Solicited-Information option to target a
    specific DODAG / version.
    """

    dodag_id: Optional[str] = None  # If set, solicits a specific DODAG

    def __post_init__(self) -> None:
        self.msg_type = MSG_DIS
        self.size_bytes = 4   # 2-byte base + minimal option overhead


@dataclass
class DIO(RPLMessage):
    """DODAG Information Object (RFC 6550 §6.3.1).

    Multicast by associated nodes to advertise DODAG topology.
    Carries the Objective Code Point so receivers can instantiate the
    correct Objective Function.
    """

    dodag_id: str = ""
    dodag_version: int = 0
    rank: int = INFINITE_RANK
    grounded: bool = True   # G flag: DODAG reaches an LBR / border router
    mop: int = 2            # Mode of Operation: 2 = Storing (no source routing)
    dtsn: int = 0           # DAO Trigger Sequence Number
    ocp: int = 0            # Objective Code Point  (0 = OF0, 1 = MRHOF)

    def __post_init__(self) -> None:
        self.msg_type = MSG_DIO
        # RFC 6550 §6.3.1: base DIO ≈ 24 bytes + DODAG-Config option ≈ 16 bytes
        self.size_bytes = 40


@dataclass
class DAO(RPLMessage):
    """Destination Advertisement Object (RFC 6550 §6.4).

    Sent upward by nodes in Storing MOP so each ancestor can cache
    downward routes.  target_prefixes is the list of /128 addresses
    reachable via the sender on this path.
    """

    dodag_id: str = ""
    dodag_version: int = 0
    target_prefixes: List[str] = field(default_factory=list)
    path_sequence: int = 0
    path_lifetime: int = 255   # 0xFF = infinite (RFC 6550 §6.4.3)
    dao_sequence: int = 0      # Matches corresponding DAO-ACK

    def __post_init__(self) -> None:
        self.msg_type = MSG_DAO
        # Base 4 bytes + 20 bytes per RPL Target option (RFC 6550 §6.7.7)
        self.size_bytes = 4 + 20 * len(self.target_prefixes)


@dataclass
class DAOAck(RPLMessage):
    """DAO Acknowledgment (RFC 6550 §6.4.2)."""

    dodag_id: str = ""
    dao_sequence: int = 0
    status: int = 0   # 0 = unconditional accept

    def __post_init__(self) -> None:
        self.msg_type = MSG_DAO_ACK
        self.size_bytes = 8
