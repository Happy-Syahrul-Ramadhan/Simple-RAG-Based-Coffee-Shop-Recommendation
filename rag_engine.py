import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/data-maps.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DAY_ORDER = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_LABELS = {
    "Mo": "Senin",
    "Tu": "Selasa",
    "We": "Rabu",
    "Th": "Kamis",
    "Fr": "Jumat",
    "Sa": "Sabtu",
    "Su": "Minggu",
}
DAY_NAME_TO_CODE = {
    "senin": "Mo",
    "selasa": "Tu",
    "rabu": "We",
    "kamis": "Th",
    "jumat": "Fr",
    "sabtu": "Sa",
    "minggu": "Su",
}
COFFEE_KEYWORDS = ("kopi", "coffee", "cafe", "kafe", "kedai kopi")
LOCATION_HINTS = (
    "bandar lampung",
    "kota bandar lampung",
    "metro",
    "kota metro",
    "lampung timur",
    "kabupaten lampung timur",
    "lampung selatan",
    "kabupaten lampung selatan",
    "lampung tengah",
    "kabupaten lampung tengah",
)
QUIET_HINTS = ("tidak ramai", "sepi", "tenang", "nggak ramai", "ga ramai", "tidak padat")
BUSY_HINTS = ("ramai", "padat", "sibuk", "penuh", "crowded")
NEAREST_HINTS = ("terdekat", "dekat", "sekitar saya", "near me", "paling dekat")
TIME_OF_DAY_HINTS = {
    "pagi": 9,
    "siang": 13,
    "sore": 16,
    "malam": 20,
}


@dataclass
class RetrievedDocument:
    text: str
    metadata: dict
    score: float


def normalize_text(value: str | None) -> str:
    return (value or "").strip()


def to_maps_url(lat: float | None, lng: float | None) -> str | None:
    if lat is None or lng is None:
        return None
    return f"https://www.google.com/maps?q={lat},{lng}"


def haversine_distance_km(
    user_lat: float | None,
    user_lng: float | None,
    place_lat: float | None,
    place_lng: float | None,
) -> float | None:
    if None in {user_lat, user_lng, place_lat, place_lng}:
        return None

    earth_radius_km = 6371.0
    lat1 = math.radians(float(user_lat))
    lon1 = math.radians(float(user_lng))
    lat2 = math.radians(float(place_lat))
    lon2 = math.radians(float(place_lng))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    }


def is_coffee_related(item: dict) -> bool:
    fields = [
        normalize_text(item.get("title")),
        normalize_text(item.get("subTitle")),
        normalize_text(item.get("categoryName")),
        " ".join(item.get("categories") or []),
    ]
    haystack = " ".join(fields).lower()
    return any(keyword in haystack for keyword in COFFEE_KEYWORDS)


def format_opening_hours(hours: list[dict] | None) -> str:
    if not hours:
        return "Jam buka tidak tersedia."
    parts = []
    for item in hours:
        day = normalize_text(item.get("day"))
        slot = normalize_text(item.get("hours"))
        if day and slot:
            parts.append(f"{day}: {slot}")
    return "; ".join(parts) if parts else "Jam buka tidak tersedia."


def summarize_popular_times(histogram: dict | None) -> str:
    if not histogram:
        return "Data jam ramai tidak tersedia."

    peaks = []
    for code in DAY_ORDER:
        buckets = histogram.get(code) or []
        if not buckets:
            continue
        top = max(buckets, key=lambda item: item.get("occupancyPercent") or 0)
        percent = top.get("occupancyPercent")
        hour = top.get("hour")
        if percent is None or hour is None or percent == 0:
            continue
        peaks.append(f"{DAY_LABELS.get(code, code)} sekitar pukul {hour:02d}.00 ({percent}%)")

    if not peaks:
        return "Data jam ramai tersedia, tetapi tidak ada puncak kunjungan yang menonjol."

    return "Puncak keramaian: " + "; ".join(peaks[:4]) + "."


def extract_query_day(query: str) -> str | None:
    query_lower = query.lower()
    for day_name, day_code in DAY_NAME_TO_CODE.items():
        if day_name in query_lower:
            return day_code
    return None


def extract_query_hour(query: str) -> int | None:
    query_lower = query.lower()
    match = re.search(r"jam\s+(\d{1,2})(?:[:.](\d{2}))?", query_lower)
    if match:
        hour = int(match.group(1))
        if "siang" in query_lower and 1 <= hour <= 5:
            return min(hour + 12, 23)
        if "sore" in query_lower and 1 <= hour <= 6:
            return min(hour + 12, 23)
        if "malam" in query_lower and 1 <= hour <= 11:
            return min(hour + 12, 23)
        return hour if 0 <= hour <= 23 else None

    for label, hour in TIME_OF_DAY_HINTS.items():
        if label in query_lower:
            return hour
    return None


def get_occupancy_percent(
    histogram: dict | None,
    hour: int,
    day_code: str | None = None,
) -> float | None:
    if not histogram:
        return None

    if day_code:
        buckets = histogram.get(day_code) or []
        for item in buckets:
            if item.get("hour") == hour:
                return item.get("occupancyPercent")
        return None

    values = []
    for code in DAY_ORDER:
        buckets = histogram.get(code) or []
        for item in buckets:
            if item.get("hour") == hour and item.get("occupancyPercent") is not None:
                values.append(float(item["occupancyPercent"]))
    if not values:
        return None
    return float(sum(values) / len(values))


def describe_occupancy(value: float | None) -> str:
    if value is None:
        return "Tidak tersedia"
    if value <= 20:
        return "Sangat sepi"
    if value <= 40:
        return "Cenderung sepi"
    if value <= 60:
        return "Sedang"
    if value <= 80:
        return "Ramai"
    return "Sangat ramai"


def format_status(item: dict) -> str:
    if item.get("permanentlyClosed"):
        return "Tempat ini permanen tutup."
    if item.get("temporarilyClosed"):
        return "Tempat ini sementara tutup."
    return "Status operasional aktif."


def parse_time_value(value: str) -> int | None:
    match = re.search(r"(\d{1,2})[.:](\d{2})", value)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    return hour * 60 + minute


def is_open_now(opening_hours: list[dict] | None, tz_name: str = "Asia/Jakarta") -> bool | None:
    if not opening_hours:
        return None

    now = datetime.now(ZoneInfo(tz_name))
    day_code = DAY_ORDER[now.weekday()]
    now_minutes = now.hour * 60 + now.minute

    for item in opening_hours:
        day_name = normalize_text(item.get("day")).lower()
        hours_text = normalize_text(item.get("hours")).lower()
        if DAY_NAME_TO_CODE.get(day_name) != day_code:
            continue
        if not hours_text:
            return None
        if "24 jam" in hours_text:
            return True
        if "tutup" in hours_text:
            return False

        matches = re.findall(r"\d{1,2}[.:]\d{2}", hours_text)
        if len(matches) < 2:
            return None

        start = parse_time_value(matches[0])
        end = parse_time_value(matches[1])
        if start is None or end is None:
            return None

        if start <= end:
            return start <= now_minutes <= end
        return now_minutes >= start or now_minutes <= end

    return None


def build_document(item: dict) -> str:
    title = normalize_text(item.get("title")) or "Nama tidak tersedia"
    category = normalize_text(item.get("categoryName")) or "Kategori tidak tersedia"
    subtitle = normalize_text(item.get("subTitle"))
    address = normalize_text(item.get("address")) or "Alamat tidak tersedia"
    neighborhood = normalize_text(item.get("neighborhood"))
    city = normalize_text(item.get("city"))
    state = normalize_text(item.get("state"))
    phone = normalize_text(item.get("phone"))
    menu = normalize_text(item.get("menu"))
    plus_code = normalize_text(item.get("plusCode"))
    live_text = normalize_text(item.get("popularTimesLiveText"))
    histogram = item.get("popularTimesHistogram")

    location = item.get("location") or {}
    lat = location.get("lat")
    lng = location.get("lng")

    score = item.get("totalScore")
    reviews = item.get("reviewsCount")
    images = item.get("imagesCount")

    parts = [
        f"Nama tempat: {title}.",
        f"Kategori utama: {category}.",
        f"Kategori tambahan: {', '.join(item.get('categories') or []) or 'Tidak tersedia'}.",
        f"Alamat lengkap: {address}.",
        f"Area: {neighborhood or 'Tidak tersedia'}.",
        f"Kota/Kabupaten: {city or 'Tidak tersedia'}.",
        f"Provinsi: {state or 'Tidak tersedia'}.",
        format_status(item),
        f"Rating Google Maps: {score if score is not None else 'Tidak tersedia'}.",
        f"Jumlah review: {reviews if reviews is not None else 'Tidak tersedia'}.",
        f"Jumlah foto: {images if images is not None else 'Tidak tersedia'}.",
        f"Jam buka: {format_opening_hours(item.get('openingHours'))}",
        f"Status keramaian saat di-scrape: {live_text or 'Tidak tersedia'}.",
        summarize_popular_times(histogram),
        (
            "Estimasi keramaian jam 13.00: "
            f"{describe_occupancy(get_occupancy_percent(histogram, 13))} "
            f"({get_occupancy_percent(histogram, 13):.1f}%)."
            if get_occupancy_percent(histogram, 13) is not None
            else "Estimasi keramaian jam 13.00 tidak tersedia."
        ),
        f"Nomor telepon: {phone or 'Tidak tersedia'}.",
        f"Menu: {menu or 'Tidak tersedia'}.",
        f"Plus code: {plus_code or 'Tidak tersedia'}.",
        f"Koordinat: {lat if lat is not None else 'Tidak tersedia'}, {lng if lng is not None else 'Tidak tersedia'}.",
    ]

    if subtitle:
        parts.insert(2, f"Subjudul atau penjelasan singkat: {subtitle}.")

    return "\n".join(parts)


class CoffeeRAG:
    def __init__(self, data_path: Path = DATA_PATH):
        raw_data = json.loads(data_path.read_text(encoding="utf-8"))
        self.records = [item for item in raw_data if is_coffee_related(item)]
        self.documents = [build_document(item) for item in self.records]
        self.metadata = [
            {
                "title": item.get("title"),
                "category": item.get("categoryName"),
                "city": item.get("city"),
                "address": item.get("address"),
                "score": item.get("totalScore"),
                "reviews_count": item.get("reviewsCount"),
                "place_id": item.get("placeId"),
                "opening_hours": format_opening_hours(item.get("openingHours")),
                "opening_hours_raw": item.get("openingHours") or [],
                "popular_live_text": item.get("popularTimesLiveText"),
                "phone": item.get("phone"),
                "popular_histogram_raw": item.get("popularTimesHistogram") or {},
                "lat": (item.get("location") or {}).get("lat"),
                "lng": (item.get("location") or {}).get("lng"),
                "maps_url": to_maps_url(
                    (item.get("location") or {}).get("lat"),
                    (item.get("location") or {}).get("lng"),
                ),
                "avg_occupancy_13": get_occupancy_percent(item.get("popularTimesHistogram"), 13),
            }
            for item in self.records
        ]
        self.doc_tokens = [tokenize(doc) for doc in self.documents]
        self.cities = sorted(
            {normalize_text(item.get("city")) for item in self.records if normalize_text(item.get("city"))}
        )

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = self.model.encode(
            self.documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        city: str | None = None,
        min_rating: float = 0.0,
        open_now: bool = False,
        user_lat: float | None = None,
        user_lng: float | None = None,
    ) -> list[RetrievedDocument]:
        query_tokens = tokenize(query)
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_embedding, dtype="float32")
        candidate_count = min(max(k * 4, 12), len(self.documents))
        scores, indices = self.index.search(query_vector, candidate_count)

        wants_best = any(term in query.lower() for term in ("terbaik", "bagus", "rekomendasi", "favorit"))
        wants_review = any(term in query.lower() for term in ("review", "ulasan", "ramai", "populer"))
        wants_quiet = any(term in query.lower() for term in QUIET_HINTS)
        wants_busy = any(term in query.lower() for term in BUSY_HINTS) and not wants_quiet
        wants_nearest = any(term in query.lower() for term in NEAREST_HINTS)
        requested_day = extract_query_day(query)
        requested_hour = extract_query_hour(query)

        rescored = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            metadata = self.metadata[idx]
            city_value = normalize_text(metadata.get("city"))
            if city and city != "Semua kota" and city_value != city:
                continue

            rating_value = metadata.get("score") or 0
            if rating_value < min_rating:
                continue

            open_status = is_open_now(metadata.get("opening_hours_raw"))
            metadata["open_now"] = open_status
            if open_now and open_status is not True:
                continue

            occupancy = None
            if requested_hour is not None:
                occupancy = get_occupancy_percent(
                    metadata.get("popular_histogram_raw"),
                    requested_hour,
                    requested_day,
                )
            metadata["requested_hour"] = requested_hour
            metadata["requested_day"] = requested_day
            metadata["requested_occupancy"] = occupancy
            metadata["requested_occupancy_label"] = describe_occupancy(occupancy)
            distance_km = haversine_distance_km(
                user_lat,
                user_lng,
                metadata.get("lat"),
                metadata.get("lng"),
            )
            metadata["distance_km"] = distance_km

            final_score = float(score)

            overlap = len(query_tokens & self.doc_tokens[idx])
            final_score += overlap * 0.02

            query_lower = query.lower()
            city_lower = city_value.lower()
            for hint in LOCATION_HINTS:
                if hint in query_lower and hint in city_lower:
                    final_score += 0.18

            reviews = metadata.get("reviews_count") or 0
            rating = rating_value
            if wants_best:
                final_score += min(rating / 25, 0.2)
                final_score += min(np.log1p(reviews) / 40, 0.08)
            if wants_review:
                final_score += min(np.log1p(reviews) / 20, 0.15)

            if requested_hour is not None:
                if occupancy is not None:
                    final_score += 0.1
                    if wants_quiet:
                        final_score += (100 - occupancy) / 120
                    elif wants_busy:
                        final_score += occupancy / 120
                    else:
                        final_score += 0.03
                else:
                    final_score -= 0.15

            if wants_quiet and occupancy is not None and occupancy <= 35:
                final_score += 0.12
            if wants_busy and occupancy is not None and occupancy >= 65:
                final_score += 0.12

            if user_lat is not None and user_lng is not None and distance_km is not None:
                final_score += 0.05
                if wants_nearest:
                    final_score += max(0, 1.5 - min(distance_km, 15) / 10)
                else:
                    final_score += max(0, 0.35 - min(distance_km, 35) / 100)
            elif wants_nearest:
                final_score -= 0.2

            rescored.append((final_score, idx))

        rescored.sort(key=lambda item: item[0], reverse=True)

        results = []
        for final_score, idx in rescored[:k]:
            results.append(
                RetrievedDocument(
                    text=self.documents[idx],
                    metadata=self.metadata[idx],
                    score=float(final_score),
                )
            )
        return results


@lru_cache(maxsize=1)
def get_rag_engine() -> CoffeeRAG:
    return CoffeeRAG()
