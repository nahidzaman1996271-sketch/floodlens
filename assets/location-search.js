const locationSearch = document.getElementById('locationSearch');
const locationInput = document.getElementById('searchInput');
const locationResults = document.getElementById('locationResults');
const locationStatus = document.getElementById('locationStatus');
const mapOnlyNotice = document.getElementById('mapOnlyNotice');
const returnToScenario = document.getElementById('returnToScenario');
const locationCopy = {
  bn: {
    input: 'স্থান, স্টেশন বা বাড়ির আইডি খুঁজুন',
    examples: 'যেমন কুয়াকাটা, সিলেট বা B0301',
    results: 'খোঁজার ফলাফল',
    searching: 'অন্যান্য স্থান খোঁজা হচ্ছে…',
    empty: 'কোনো স্থান পাওয়া যায়নি। অন্য বানান বা কাছের শহরের নাম দিয়ে চেষ্টা করুন।',
    unavailable: 'স্থানগুলোর তালিকা লোড হয়নি। নমুনা স্টেশন, বাড়ি ও আশ্রয়কেন্দ্র খোঁজা যাবে।',
    station: 'সিমুলেশন স্টেশন',
    home: 'নমুনা বাড়ি',
    shelter: 'নমুনা আশ্রয়কেন্দ্র',
    place: 'শুধু মানচিত্র',
    mapOnly: 'এই স্থানের জন্য বন্যা সিমুলেশন নেই।',
    back: 'সিমুলেশনে ফিরুন'
  },
  en: {
    input: 'Search places, stations, or household IDs',
    examples: 'Try Kuakata, Sylhet, or B0301',
    results: 'Location results',
    searching: 'Looking up other places…',
    empty: 'No matching place. Try another spelling or a nearby town.',
    unavailable: 'The place list could not load. Demo stations, homes, and shelters are still searchable.',
    station: 'Simulation station',
    home: 'Sample home',
    shelter: 'Sample shelter',
    place: 'Map only',
    mapOnly: 'Flood simulation is unavailable for this place.',
    back: 'Return to simulation'
  }
};

let locationIndexPromise;
let locationRevision = 0;
let browseMarker = null;
let browsePlace = null;

function normalizeLocationName(name) {
  return name.normalize('NFD')
    .replace(/(\p{Script=Latin})\p{M}+/gu, '$1')
    .replace(/[\u200c\u200d]/g, '')
    .replace(/[০-৯]/g, digit => String(digit.charCodeAt(0) - 0x09e6))
    .replace(/[^\p{L}\p{M}\p{N}]+/gu, ' ')
    .trim().toLowerCase();
}

function searchableNames(names) {
  return [...new Set(names.filter(Boolean).map(normalizeLocationName))];
}

const demoLocations = Object.entries(STATIONS).flatMap(([key, station]) => [
  {
    kind: 'station', key, name: station.nameEn, bn: station.name,
    terms: searchableNames([
      key, station.name, station.nameEn, station.name.split(' (')[0], station.nameEn.split(' (')[0],
      station.localityName, station.localityNameEn,
      key === 'sylhet' && 'সিলেট'
    ])
  },
  ...[
    ...station.households.map(home => ({ item: home, kind: 'home' })),
    ...station.shelters.map(shelter => ({ item: shelter, kind: 'shelter' }))
  ].map(({ item, kind }) => ({
    kind, key, id: item.id, name: item.nameEn, bn: item.name,
    terms: searchableNames([item.nameEn, item.name, item.id])
  }))
]);

function getPlaceIndex() {
  if (!locationIndexPromise) {
    locationIndexPromise = fetch('./assets/bangladesh-places.json')
      .then(response => {
        if (!response.ok) throw new Error('Place index unavailable');
        return response.json();
      })
      .then(({ districts, places }) => places.map(([name, bn, lat, lon, district, ...aliases]) => ({
        kind: 'place', name, bn, coords: [lat, lon], district: districts[district],
        terms: searchableNames([name, bn, ...aliases])
      })))
      .catch(error => {
        locationIndexPromise = undefined;
        throw error;
      });
  }
  return locationIndexPromise;
}

function matchingLocations(entries, query, limit) {
  const matches = [];
  for (const entry of entries) {
    const score = entry.terms.reduce((best, name) => {
      if (name === query) return 3;
      if (name.startsWith(query)) return Math.max(best, 2);
      if (name.includes(query)) return Math.max(best, 1);
      return best;
    }, 0);
    if (!score) continue;
    const position = matches.findIndex(match => score > match.score);
    if (position === -1) {
      if (matches.length < limit) matches.push({ entry, score });
    } else {
      matches.splice(position, 0, { entry, score });
      if (matches.length > limit) matches.pop();
    }
  }
  return matches;
}

function locationName(location) {
  return currentLang === 'bn' && location.bn ? location.bn : location.name;
}

function renderLocationResults(matches, status = '') {
  const copy = locationCopy[currentLang];
  locationStatus.textContent = status;
  const focusedIndex = [...locationResults.children].findIndex(row => row.contains(document.activeElement));
  const rows = matches.map(({ entry }) => {
    const row = document.createElement('li');
    const button = document.createElement('button');
    const title = document.createElement('span');
    const detail = document.createElement('small');
    const district = entry.district?.[currentLang === 'bn' ? 1 : 0] || entry.district?.[0];
    const station = entry.key && (currentLang === 'bn' ? STATIONS[entry.key].name : STATIONS[entry.key].nameEn);

    button.type = 'button';
    title.textContent = locationName(entry);
    detail.textContent = [copy[entry.kind], entry.id, entry.kind !== 'station' && (station || district)]
      .filter(Boolean).join(' · ');
    button.append(title, detail);
    button.addEventListener('click', () => selectLocation(entry));
    row.append(button);
    return row;
  });
  locationResults.replaceChildren(...rows);
  locationResults.hidden = !rows.length;
  if (focusedIndex >= 0) (rows[Math.min(focusedIndex, rows.length - 1)]?.firstElementChild || locationInput).focus();
}

async function updateLocationResults() {
  const revision = ++locationRevision;
  const query = normalizeLocationName(locationInput.value);
  const copy = locationCopy[currentLang];
  if (query.length < 2) {
    renderLocationResults([]);
    return;
  }
  const nearby = matchingLocations(demoLocations, query, 6);
  renderLocationResults(nearby, copy.searching);
  try {
    const places = await getPlaceIndex();
    if (revision !== locationRevision) return;
    const others = matchingLocations(places, query, 6);
    const matches = [...nearby, ...others]
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
    renderLocationResults(matches, matches.length ? '' : copy.empty);
  } catch {
    if (revision === locationRevision) renderLocationResults(nearby, copy.unavailable);
  }
}

function closeLocationResults() {
  locationRevision++;
  renderLocationResults([]);
}

function refreshLocationSearchLanguage() {
  const copy = locationCopy[currentLang];
  locationInput.placeholder = copy.examples;
  locationInput.setAttribute('aria-label', copy.input);
  locationResults.setAttribute('aria-label', copy.results);
  returnToScenario.textContent = copy.back;
  if (browsePlace) updatePlaceNotice();
  if (locationInput.value && locationSearch.contains(document.activeElement)) updateLocationResults();
}

function updatePlaceNotice() {
  const copy = locationCopy[currentLang];
  const title = document.createElement('strong');
  const note = document.createElement('span');
  title.textContent = locationName(browsePlace);
  note.textContent = copy.mapOnly;
  document.getElementById('mapOnlyText').replaceChildren(title, note);
  mapOnlyNotice.hidden = false;
}

function showMapPlace(location) {
  if (isPlaying) togglePlaySimulation();
  floodLayerGroup.clearLayers();
  markersLayerGroup.clearLayers();
  homePoints.clear();
  if (routePolyline) map.removeLayer(routePolyline);
  if (browseMarker) map.removeLayer(browseMarker);

  browsePlace = location;
  document.body.classList.add('browsing-place');
  updatePlaceNotice();
  map.invalidateSize({ pan: false });
  map.setView(location.coords, 13, { animate: false });
  browseMarker = L.circleMarker(location.coords, {
    radius: 7, color: '#fcfcf8', weight: 2, fillColor: '#215b78',
    fillOpacity: 1, interactive: false
  }).addTo(map);
}

function leavePlaceBrowse() {
  if (!browsePlace) return;
  map.removeLayer(browseMarker);
  browseMarker = null;
  browsePlace = null;
  document.body.classList.remove('browsing-place');
  mapOnlyNotice.hidden = true;
  map.invalidateSize({ pan: false });
}

function selectLocation(location) {
  locationInput.value = locationName(location);
  closeLocationResults();
  locationInput.focus();
  if (location.kind === 'place') {
    showMapPlace(location);
    return;
  }
  if (browsePlace || activeStation !== STATIONS[location.key] || location.kind === 'station') {
    changeStation(location.key);
  }
  if (location.id) quickSelect(location.id);
}

locationInput.addEventListener('input', updateLocationResults);
locationSearch.addEventListener('submit', event => {
  event.preventDefault();
  locationResults.querySelector('button')?.click();
});
locationSearch.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    closeLocationResults();
    locationInput.focus();
  } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    const buttons = [...locationResults.querySelectorAll('button')];
    if (!buttons.length) return;
    const current = buttons.indexOf(document.activeElement);
    const next = current === -1 ? 0 : current + (event.key === 'ArrowDown' ? 1 : -1);
    (buttons[next] || locationInput).focus();
    event.preventDefault();
  }
});
locationSearch.addEventListener('focusout', event => {
  if (!locationSearch.contains(event.relatedTarget)) closeLocationResults();
});
returnToScenario.addEventListener('click', () => {
  changeStation(document.getElementById('stationSelect').value);
  locationInput.value = '';
  locationInput.focus();
});
refreshLocationSearchLanguage();
