import { GoogleGenerativeAI, SchemaType } from '@google/generative-ai';
import type { Schema } from '@google/generative-ai';
import type { NextRequest } from 'next/server';
import type { AnalysisResult } from '../../lib/types';

interface RequestBody {
  skin_hex: string;
  eye_hex: string;
  hair_hex: string;
}

// Strict JSON schema — cast via unknown because TypeScript widens enum members
// in object literals (SchemaType.OBJECT inferred as SchemaType, not its literal).
const responseSchema = {
  type: SchemaType.OBJECT,
  properties: {
    season:           { type: SchemaType.STRING },
    description:      { type: SchemaType.STRING },
    undertone:        { type: SchemaType.STRING },
    contrast:         { type: SchemaType.STRING },
    metal:            { type: SchemaType.STRING },
    best_colors:      { type: SchemaType.ARRAY, items: { type: SchemaType.STRING } },
    best_hair_colors: { type: SchemaType.ARRAY, items: { type: SchemaType.STRING } },
    avoid_colors:     { type: SchemaType.ARRAY, items: { type: SchemaType.STRING } },
  },
  required: ['season', 'description', 'undertone', 'contrast', 'metal', 'best_colors', 'best_hair_colors', 'avoid_colors'],
} as unknown as Schema;

function isValidHex(s: string) {
  return typeof s === 'string' && /^#[0-9a-fA-F]{6}$/.test(s);
}

export async function POST(request: NextRequest) {
  const apiKey = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return Response.json({ error: 'GOOGLE_API_KEY (or GEMINI_API_KEY) not configured' }, { status: 500 });
  }

  let body: Partial<RequestBody>;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { skin_hex, eye_hex, hair_hex } = body;
  if (!isValidHex(skin_hex!) || !isValidHex(eye_hex!) || !isValidHex(hair_hex!)) {
    return Response.json(
      { error: 'skin_hex, eye_hex, and hair_hex must be valid 6-digit hex codes' },
      { status: 400 }
    );
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel(
    {
      model: process.env.GEMINI_MODEL ?? 'gemini-2.5-flash-lite',
      generationConfig: {
        responseMimeType: 'application/json',
        responseSchema,
      },
    },
    { apiVersion: 'v1beta' }
  );

  const prompt = `You are an expert color season analyst with deep knowledge of the 12-season system.

── SEASON SIGNATURES ───────────────────────────────────────────────────────
SPRING family (warm undertone, clear/vivid coloring):
  • Light Spring  – very light depth (skin lum > 180), delicate warmth, low contrast; fairest Spring
  • Warm Spring   – strongest golden warmth, medium depth, clear peachy-golden glow
  • True Spring   – warm, medium depth, clear and bright; balanced spring

SUMMER family (cool undertone, muted/soft coloring):
  • Light Summer  – very light depth (skin lum > 175), cool-rose/pink undertone, low contrast; fairest Summer
  • Soft Summer   – cool-neutral, medium-low depth, very dusty/muted; warmest Summer
  • True Summer   – cool, medium depth, soft rose-pink undertone; most clearly cool Summer

AUTUMN family (warm undertone, muted/earthy coloring):
  • Soft Autumn   – warm-neutral, light-to-medium depth, very muted/dusty; warmest/softest Autumn
  • Warm Autumn   – strong golden/amber undertone, MEDIUM depth (hair lum 90–150), richly warm; hair is medium-to-dark brown or auburn, NOT near-black
  • Deep Autumn   – warm undertone, DEEP depth (hair lum < 90 AND eye lum < 105), medium contrast; darkest Autumn — hair is dark brown to near-black, eyes are dark

WINTER family (cool undertone, clear coloring):
  • True Winter   – cool, HIGH contrast between light skin and dark hair/eyes, vivid
  • Cool Winter   – strongly cool/blue-pink undertone, medium-high depth, clear
  • Dark Winter   – cool-to-neutral, DEEP depth (hair lum < 85, dark eyes), very high contrast

── STEP-BY-STEP DIAGNOSIS ──────────────────────────────────────────────────
Execute each step in order. Show your explicit numeric reasoning for each step.

STEP 1 — Compute luminance (L = 0.299×R + 0.587×G + 0.114×B, values 0–255):
  • Compute skin_L, eye_L, hair_L and write down the numbers.

STEP 2 — Determine DEPTH tier from hair_L (hair is the dominant depth signal):
  • hair_L < 90  → DARK-HAIR (black to dark brown)
  • hair_L 90–145 → MEDIUM-HAIR (medium brown to dark blonde)
  • hair_L > 145 → LIGHT-HAIR (blonde / light brown)

  Then refine using eye_L:
  • DARK-HAIR + eye_L < 105 → DEEP  (candidates: Deep Autumn, Dark Winter)
  • DARK-HAIR + eye_L ≥ 105 → MEDIUM-DARK
  • MEDIUM-HAIR → MEDIUM
  • LIGHT-HAIR  → LIGHT (skin_L > 175 confirms VERY LIGHT)

STEP 3 — Determine UNDERTONE from skin hex (write out R, G, B values):
  Warm  (→ Spring or Autumn family):
    • R is largest channel AND (R − B) > 18
    • Skin reads peachy, golden, olive, or golden-brown
  Cool  (→ Summer or Winter family):
    • (R − B) < 12 OR B is close to/larger than G, skin reads pink, rosy, or ashy
  Neutral:
    • (R − B) between 12–18 with no strong cast — use hair and eye warmth to break tie

STEP 4 — DEEP tier decision (run this BEFORE routing to medium seasons):
  !! If depth = DEEP, do not proceed to STEP 5 — resolve here:
  • Warm + DEEP → Deep Autumn  (warm/olive/golden-brown skin, dark-brown-to-black hair, dark eyes)
  • Cool or Neutral + DEEP → Dark Winter  (cooler/ashy skin, near-black hair, very high contrast)
  • Ambiguous undertone + DEEP: if skin (R−B) > 15 → Deep Autumn; else → Dark Winter

  !! MEDIUM-DARK (dark hair but lighter eyes) — also resolve here:
  • Warm + MEDIUM-DARK → Deep Autumn if hair_L < 80; Warm Autumn if hair_L 80–90
  • Cool + MEDIUM-DARK → Cool Winter or True Winter (go to STEP 5C)

STEP 5 — Medium/Light tier decisions:

STEP 5A — Warm + MEDIUM: Warm Autumn vs Warm Spring
  • Warm Autumn:  hair_L 90–130, skin muted/earthy/olive (yellow-green bias, lower chroma)
  • Warm Spring:  hair_L 100–145, skin fresh/peachy/golden (pink-peach bias, higher chroma)
  • Tiebreaker: skin (R−B) > 38 AND skin_L > 152 → Warm Spring; else → Warm Autumn

STEP 5B — Warm + LIGHT: Light Spring vs Soft Autumn
  • skin_L > 182 and hair golden/blonde → Light Spring
  • skin muted warm-beige → Soft Autumn

STEP 5C — Cool + MEDIUM or MEDIUM-DARK: True Summer vs Cool Winter vs True Winter
  • (skin_L − hair_L) > 85 → True Winter (very high contrast)
  • (skin_L − hair_L) 60–85 + clear vivid features → Cool Winter
  • (skin_L − hair_L) < 60 + soft/muted features → True Summer

STEP 5D — Cool/Neutral + LIGHT: Light Summer vs Light Spring
  • Pink/rose bias in skin → Light Summer
  • Peachy/golden bias → Light Spring

STEP 5E — Neutral + MEDIUM: Soft Autumn vs Soft Summer
  • Warm-beige/golden-taupe skin, warm brown hair → Soft Autumn
  • Cool-beige/rose-taupe skin, ash brown hair → Soft Summer

── INPUT DATA ──────────────────────────────────────────────────────────────
Measured color data extracted from the person's face via computer vision:
  Skin tone : ${skin_hex}
  Eye color : ${eye_hex}
  Hair color: ${hair_hex}

── SEASON COLOR PALETTES — best_colors hex anchors ─────────────────────────
Return exactly 5 hex codes. Must span ≥ 3 distinct hue families. No 5 similar neutrals.

  Deep Autumn   : #3d1f0f (espresso), #7a2030 (burgundy), #2d5a3d (forest green), #8b3a1a (rust), #5c7a3e (olive)
  Warm Autumn   : #b85c38 (terracotta), #c87941 (pumpkin), #c9a87c (camel), #6b7a3e (moss), #8b6914 (golden)
  Soft Autumn   : #c4967a (dusty peach), #8a9a72 (sage), #b09080 (warm taupe), #c8a85a (soft gold), #7a9088 (grey-green)
  True Autumn   : #c43820 (tomato), #b86030 (burnt orange), #a07820 (dark gold), #6b7848 (khaki), #8b2818 (warm rust)
  Dark Winter   : #0a0a0f (near black), #5a0a30 (deep wine), #1a3a5a (dark teal), #3a0a5a (deep purple), #8a1030 (dark burgundy)
  True Winter   : #0a0a0a (black), #f5f5f8 (pure white), #1a2a8a (royal blue), #c00050 (fuchsia), #c80020 (true red)
  Cool Winter   : #c8d8f0 (icy blue), #c80020 (true red), #303840 (charcoal), #5020a0 (deep purple), #0030c0 (royal blue)
  Light Spring  : #f5b090 (peach), #f07860 (soft coral), #f8f0d8 (warm white), #80d8c8 (light aqua), #f0d040 (golden yellow)
  Warm Spring   : #f06040 (bright coral), #f0c020 (warm yellow), #f08000 (bright orange), #40c0b0 (warm turquoise), #f09030 (golden)
  True Spring   : #f04830 (clear coral), #50b840 (warm green), #f0c828 (golden yellow), #28b8b0 (turquoise), #f07830 (warm orange)
  Light Summer  : #c0a8d8 (lavender), #90b8d8 (soft blue), #f0c8d8 (powder pink), #9ab0c8 (grey-blue), #e8e8f0 (soft white)
  Soft Summer   : #c09898 (dusty rose), #9898c0 (muted lavender), #709898 (soft teal), #909898 (warm grey), #a890a0 (dusty mauve)
  True Summer   : #304870 (soft navy), #b03060 (raspberry), #d070a0 (cool pink), #4070a0 (medium blue), #787890 (soft grey)

── FLATTERING HAIR COLORS — best_hair_colors hex anchors ───────────────────
Return exactly 4 hex codes. Include range from natural base to bolder option. No 4 identical browns.

  Deep Autumn   : #1a0a05 (espresso), #3d1510 (dark mahogany), #5a1f00 (warm dark auburn), #6b2d10 (deep copper-brown)
  Warm Autumn   : #5a3018 (warm medium brown), #6b2010 (rich auburn), #8b3d10 (copper), #7a5020 (golden-brown)
  Soft Autumn   : #7a5838 (soft warm brown), #8a6838 (muted golden brown), #787060 (warm ash brown), #8a4828 (light copper)
  True Autumn   : #5a2810 (warm chestnut), #5a1808 (dark auburn), #7a3010 (burnt copper), #6a3820 (rich medium brown)
  Dark Winter   : #050508 (blue-black), #100808 (dark espresso), #1a1020 (deep cool brown), #2a0818 (dark burgundy tint)
  True Winter   : #050508 (true black), #050510 (blue-black), #1a1020 (dark cool brown), #e8e8f0 (stark platinum)
  Cool Winter   : #080808 (true black), #181820 (dark ash brown), #201818 (cool dark brown), #d8d8e0 (platinum)
  Light Spring  : #e8c890 (warm light blonde), #c89840 (honey blonde), #c87860 (strawberry blonde), #9a6838 (light warm brown)
  Warm Spring   : #d8b040 (golden blonde), #a84020 (bright copper), #d09828 (warm honey), #8b3818 (light warm auburn)
  True Spring   : #d8b848 (clear warm blonde), #d0a820 (bright golden), #9a6830 (warm light brown), #a83e18 (light copper)
  Light Summer  : #c8c0a0 (ash blonde), #a09080 (cool light brown), #786858 (cool medium brown), #d0d0d8 (platinum ash)
  Soft Summer   : #908070 (muted ash brown), #807060 (cool medium brown), #b0a888 (soft warm-grey blonde), #786858 (dusty brown)
  True Summer   : #786858 (cool medium brown), #887870 (ash brown), #a09080 (soft cool dark blonde), #080810 (blue-toned black)

── COLORS TO AVOID — avoid_colors hex anchors ──────────────────────────────
Return exactly 3 hex codes. CRITICAL RULES:
  1. The 3 colors MUST come from at least 3 DIFFERENT hue families — e.g. one cool purple, one warm orange, one pale icy tone. NEVER return 3 similar warm-orange shades or 3 similar greys.
  2. Each color must be visually and chromatically distinct from the others (spread across the color wheel).
  3. Choose the most dramatically wrong colors — the ones that would most clash with the season.

Season avoid anchors (one from each distinct hue family — pick these or close variants):
  Deep Autumn   : #c8a0e0 (icy lavender), #a8c8f0 (baby blue), #80ff40 (neon lime green)
  Warm Autumn   : #9880c8 (cool lavender), #f4b8d0 (icy pink), #506880 (slate blue-grey)
  Soft Autumn   : #0a0a0a (stark black), #a0d8f0 (icy pale blue), #c0f000 (neon yellow-green)
  True Autumn   : #c0a0e0 (icy lavender), #d0e820 (neon yellow), #6080a8 (cool blue-grey)
  Dark Winter   : #c89858 (warm camel), #e8b830 (golden yellow), #c07040 (warm terracotta)
  True Winter   : #a07840 (warm brown), #c09858 (warm camel), #b09080 (dusty warm taupe)
  Cool Winter   : #b86030 (warm rust), #c89858 (warm camel), #d4a030 (golden yellow)
  Light Spring  : #404850 (cool charcoal), #a090c0 (cool lavender), #0a0a0a (stark black)
  Warm Spring   : #7080a8 (cool blue-grey), #c0b0d8 (icy lavender), #808080 (cool grey)
  True Spring   : #808890 (cool grey), #907890 (dusty mauve), #c0d0c0 (muted cool sage)
  Light Summer  : #c07040 (warm rust), #d4a830 (golden yellow), #c07830 (warm copper)
  Soft Summer   : #ff5020 (neon orange-red), #d4a020 (warm golden), #0a0a0a (stark black)
  True Summer   : #b85820 (warm rust), #c89858 (warm camel), #d07820 (golden orange)

── OUTPUT FIELDS ────────────────────────────────────────────────────────────
• "season"           – exact season name from the 12 seasons above
• "description"      – 2–3 sentences citing the computed luminance values and undertone rationale
• "undertone"        – exactly one word: "Warm", "Cool", or "Neutral"
• "contrast"         – exactly one word: "High", "Medium", or "Low"
• "metal"            – best jewelry metal: "Gold", "Silver", "Rose Gold", or "Both"
• "best_colors"      – exactly 5 hex codes from the season's palette anchors above; span ≥ 3 hue families
• "best_hair_colors" – exactly 4 hex codes from the season's hair anchors above; range from base to bold
• "avoid_colors"     – exactly 3 hex codes from the season's avoid anchors above; MUST be from 3 different hue families and visually distinct from each other

All hex codes must be 6-digit lowercase (e.g. #a3c4e2).`;

  try {
    const result = await model.generateContent(prompt);
    const json: AnalysisResult = JSON.parse(result.response.text());
    return Response.json(json);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[analyze-color]', msg);
    return Response.json({ error: msg }, { status: 500 });
  }
}
