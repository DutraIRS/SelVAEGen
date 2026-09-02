import numpy as np
import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

RDLogger.DisableLog('rdApp.*')
_pharm_factory = Gobbi_Pharm2D.factory

ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                'Zr', 'Cr', 'Pt', 'Hg', 'Pb']

HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2]

FORMAL_CHARGES = [-1, -2, 1, 2, 0]

CHIRAL_TAGS = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]

BOND_STEREO = [Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE]

BOND_TYPES = [None] + [
    (order, stereo)
    for order in (Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC)
    for stereo in BOND_STEREO
]

DEGREES = list(range(9))
HYDROGENS = list(range(9))
VALENCES = list(range(9))

BOND_NUM_FEATURES = 6
NUM_BOND_STEREO = len(BOND_STEREO)

ATOM_BLOCKS = (
    ("symbol", len(ATOM_SYMBOLS) + 1),
    ("degree", len(DEGREES) + 1),
    ("num_hs", len(HYDROGENS) + 1),
    ("valence", len(VALENCES) + 1),
    ("charge", len(FORMAL_CHARGES) + 1),
    ("hybridization", len(HYBRIDIZATIONS) + 1),
    ("chiral", len(CHIRAL_TAGS) + 1),
)

DECODED_ATOM_BLOCKS = tuple((name, size) for name, size in ATOM_BLOCKS if name in ("symbol", "charge", "chiral"))

# the two booleans closing the binary block, then the two continuous features
ATOM_FLAGS = ("aromatic", "in_ring")
ATOM_EXTRAS = ("atomic_num", "radical_electrons")

# feature widths follow from the block definitions above
ATOM_BINARY_FEATURES = sum(size for _, size in ATOM_BLOCKS) + len(ATOM_FLAGS)
ATOM_NUM_FEATURES = ATOM_BINARY_FEATURES + len(ATOM_EXTRAS)

NUM_BOND_TYPES = 4


def one_hot_encoding(x, allowable_set):
    unknown = x not in allowable_set
    return [x == s for s in allowable_set] + [unknown]

def atom_features(atom):
    binary = np.array(
        one_hot_encoding(atom.GetSymbol(), ATOM_SYMBOLS) +
        one_hot_encoding(atom.GetDegree(), DEGREES) +
        one_hot_encoding(atom.GetTotalNumHs(), HYDROGENS) +
        one_hot_encoding(atom.GetTotalValence(), VALENCES) +
        one_hot_encoding(atom.GetFormalCharge(), FORMAL_CHARGES) +
        one_hot_encoding(atom.GetHybridization(), HYBRIDIZATIONS) +
        one_hot_encoding(atom.GetChiralTag(), CHIRAL_TAGS) +
        [atom.GetIsAromatic(), atom.IsInRing()],
        dtype=np.float32)

    extra = np.array([atom.GetAtomicNum() / 100.0, atom.GetNumRadicalElectrons()], dtype=np.float32)

    return np.concatenate([binary, extra])

def bond_features(bond):
    bond_type = bond.GetBondType()
    stereo = bond.GetStereo()
    return [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetBondTypeAsDouble(),
        BOND_STEREO.index(stereo) if stereo in BOND_STEREO else 0,
    ]

def make_graph(smiles, device='cpu'):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")

    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)

    nodes = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.array(nodes), dtype=torch.float32, device=device)

    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges.extend([(i, j), (j, i)])
        edge_attrs.extend([bf, bf])

    if edges:
        edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, BOND_NUM_FEATURES), dtype=torch.float32, device=device)

    return x, edge_index, edge_attr

def split_atom_features(x):
    """Split node features into the blocks written by atom_features."""
    blocks, offset = {}, 0

    for name, size in ATOM_BLOCKS:
        blocks[name] = x[..., offset:offset + size]
        offset += size

    blocks["flags"] = x[..., offset:offset + len(ATOM_FLAGS)]
    blocks["extras"] = x[..., offset + len(ATOM_FLAGS):]

    return blocks

@torch.no_grad()
def dense_bond_types(data_batch, num_nodes):
    """Dense [batch, nodes, nodes] view of make_graph edges, as indices into BOND_TYPES."""
    batch = data_batch.batch
    counts = torch.bincount(batch)
    offsets = torch.cumsum(counts, 0) - counts  # first global index of each graph

    source, target = data_batch.edge_index
    graph = batch[source]

    # edge_attr is [single, double, triple, aromatic, orderAsDouble, stereo], the last
    # column indexing BOND_STEREO, so class = 1 + 3 * order_idx + stereo_idx
    order_idx = data_batch.edge_attr[:, :NUM_BOND_TYPES].argmax(-1)
    stereo = data_batch.edge_attr[:, -1].round().long().clamp(0, NUM_BOND_STEREO - 1)
    kind = 1 + NUM_BOND_STEREO * order_idx + stereo

    types = torch.zeros(counts.numel(), num_nodes, num_nodes, dtype=torch.long, device=batch.device)
    types[graph, source - offsets[graph], target - offsets[graph]] = kind

    return types

def to_mols(smiles):
    """Parse SMILES, None wherever the string is empty or does not parse."""
    return [Chem.MolFromSmiles(s) if s else None for s in smiles]

def safe_smiles(mol, canonical=True):
    """The SMILES for a molecule, or None if RDKit cannot write one."""
    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=canonical)
    except Exception:
        return None

def compute_fingerprints(smiles, fp_dim=512, device='cpu'):
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_dim)
    morgan = torch.tensor(np.asarray(fp, dtype=np.float32), device=device)

    fp = MACCSkeys.GenMACCSKeys(mol)
    maccs = torch.tensor(np.asarray(fp, dtype=np.float32), device=device)

    fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=fp_dim)
    daylight = torch.tensor(np.asarray(fp, dtype=np.float32), device=device)

    fp = Generate.Gen2DFingerprint(mol, _pharm_factory)
    pharmacophore = torch.zeros(fp_dim, dtype=torch.float32, device=device)
    on_bits = torch.tensor(list(fp.GetOnBits()), dtype=torch.long, device=device)

    if len(on_bits):
        pharmacophore[on_bits % fp_dim] = 1.0

    return torch.cat([morgan, maccs, daylight, pharmacophore])


ORDER_VALENCE = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2,
                    Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 1}


# a positive charge buys a group 15 atom one more bond, anywhere else a charge costs one
GAINS_VALENCE = ("N", "P", "As", "Sb")

# stand-in budget for the elements RDKit assigns no default valence, so they can still bond
UNCAPPED_VALENCE = 1 << 30

# tried in order, in fixed four-slot positions so assembly_tier always means the same
ASSEMBLY_TIERS = ((True, False), (True, True), (False, False), (False, True))


def _apply_bond_stereo(molecule, wanted):
    """Set E/Z on the double bonds that asked for it, once every neighbour exists."""
    for (i, j), stereo in wanted.items():
        bond = molecule.GetBondBetweenAtoms(i, j)

        if bond is None or bond.GetBondType() != Chem.rdchem.BondType.DOUBLE:
            continue

        begin = [a.GetIdx() for a in bond.GetBeginAtom().GetNeighbors() if a.GetIdx() != j]
        end = [a.GetIdx() for a in bond.GetEndAtom().GetNeighbors() if a.GetIdx() != i]

        if not begin or not end:
            continue                    # a terminal double bond has no geometry to set

        bond.SetStereoAtoms(begin[0], end[0])
        bond.SetStereo(stereo)

def _build(symbols, charges, chiral, bonds, demote_aromatic):
    """One assembly attempt: the largest sanitised fragment, or None if RDKit refuses."""
    table = Chem.GetPeriodicTable()
    mol, budget = Chem.RWMol(), []

    for position, symbol in enumerate(symbols):
        atom = Chem.Atom(symbol)
        charge = 0

        if charges is not None and charges[position] < len(FORMAL_CHARGES):
            charge = FORMAL_CHARGES[charges[position]]
            atom.SetFormalCharge(charge)
        if chiral is not None and chiral[position] < len(CHIRAL_TAGS):
            atom.SetChiralTag(CHIRAL_TAGS[chiral[position]])

        mol.AddAtom(atom)
        allowed = table.GetDefaultValence(table.GetAtomicNumber(symbol))
        adjustment = charge if symbol in GAINS_VALENCE else -abs(charge)

        # RDKit reports -1 for the elements it never valence-checks, so neither does this
        budget.append(UNCAPPED_VALENCE if allowed < 0 else max(allowed + adjustment, 0))

    placed, wanted = 0, {}
    for _, i, j, cost, order, stereo in bonds:
        if budget[i] >= cost and budget[j] >= cost:
            aromatic = order == Chem.rdchem.BondType.AROMATIC
            mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE if aromatic and demote_aromatic else order)
            budget[i] -= cost
            budget[j] -= cost
            placed += 1

            if stereo != Chem.rdchem.BondStereo.STEREONONE and not (aromatic and demote_aromatic):
                wanted[(i, j)] = stereo

    try:
        molecule = mol.GetMol()
        Chem.SanitizeMol(molecule)
        _apply_bond_stereo(molecule, wanted)

        pieces = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
        if not pieces:
            return None

        largest = max(pieces, key=lambda piece: piece.GetNumAtoms())
        Chem.SanitizeMol(largest)
    except Exception:
        return None

    # what the assembler had to throw away to get here, so a metric can say so
    largest.SetIntProp("bonds_skipped", len(bonds) - placed)
    largest.SetIntProp("atoms_dropped", len(symbols) - largest.GetNumAtoms())

    return largest

def assemble_molecule(symbols, bond_scores, chiral=None, charges=None):
    """Greedy assembly under a per-atom valence budget, so the result always sanitises."""
    if not symbols:
        return None

    scores = np.asarray(bond_scores, dtype=np.float64)
    scores = (scores + scores.transpose(1, 0, 2)) / 2      # the pair matrix is symmetric

    candidates = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pair = scores[i, j]
            best = int(pair[1:].argmax()) + 1

            if pair[best] <= pair[0]:
                continue                                    # the model prefers no bond here

            order, stereo = BOND_TYPES[best]
            candidates.append((float(pair[best]), i, j, ORDER_VALENCE[order], order, stereo))

    bonds = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))

    for tier, (attempt_charges, demote) in enumerate(ASSEMBLY_TIERS):
        if attempt_charges and charges is None:
            continue                                    # nothing was predicted to keep

        molecule = _build(symbols, charges if attempt_charges else None, chiral, bonds, demote)
        if molecule is not None:
            molecule.SetIntProp("assembly_tier", tier)
            molecule.SetIntProp("charges_dropped", int(charges is not None and not attempt_charges))
            molecule.SetIntProp("aromatics_demoted", int(demote))
            return molecule

    return None
