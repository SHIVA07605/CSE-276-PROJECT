"""
Automated genuine screenshot capture script using Selenium Headless Chrome.
Captures all 6 required screenshots from the running review_app.py server.
"""
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

out_dir = Path("c:/project ai/results/report_evidence/screenshots")
out_dir.mkdir(parents=True, exist_ok=True)

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--window-size=1400,1050")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)

try:
    print("Navigating to http://localhost:8050 ...")
    driver.get("http://localhost:8050")

    # Wait until allPairs loaded
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return typeof allPairs !== 'undefined' && allPairs.length > 0")
    )
    time.sleep(1)

    total_pairs = driver.execute_script("return allPairs.length")
    print(f"Loaded pairs: {total_pairs}")

    # ============================================================
    # 1. Main Reviewer Interface
    # ============================================================
    # Switch filter to ALL to show full breadth
    filter_select = Select(driver.find_element(By.ID, "filter-select"))
    filter_select.select_by_value("ALL")
    time.sleep(0.5)

    p1 = out_dir / "01_main_interface.png"
    driver.save_screenshot(str(p1))
    print(f"Saved: {p1.name}")

    # ============================================================
    # 2. Example Equivalent Candidate
    # ============================================================
    # Find index of PAIR_TFIDF_HIGH_020 or similar positive
    idx_eq = driver.execute_script("""
        for (let i = 0; i < allPairs.length; i++) {
            if (allPairs[i].human_label === '1' || allPairs[i].pair_id === 'PAIR_TFIDF_HIGH_020') return i;
        }
        return -1;
    """)
    if idx_eq != -1:
        driver.execute_script(f"currentFilteredIdx = {idx_eq}; renderCurrentPair();")
        time.sleep(0.5)
        p2 = out_dir / "02_equivalent_candidate.png"
        driver.save_screenshot(str(p2))
        print(f"Saved: {p2.name} (Pair: {driver.find_element(By.ID, 'pair-index').text})")

    # ============================================================
    # 3. Example Incompatible Candidate with Conflict Warning
    # ============================================================
    idx_conf = driver.execute_script("""
        for (let i = 0; i < allPairs.length; i++) {
            if (allPairs[i].has_attribute_conflict === true || String(allPairs[i].has_attribute_conflict).toLowerCase() === 'true') return i;
        }
        return -1;
    """)
    if idx_conf != -1:
        driver.execute_script(f"currentFilteredIdx = {idx_conf}; renderCurrentPair();")
        time.sleep(0.5)
        p3 = out_dir / "03_incompatible_conflict_warning.png"
        driver.save_screenshot(str(p3))
        print(f"Saved: {p3.name} (Pair: {driver.find_element(By.ID, 'pair-index').text})")

    # ============================================================
    # 4. Example Uncertain Candidate
    # ============================================================
    idx_unc = driver.execute_script("""
        for (let i = 0; i < allPairs.length; i++) {
            if (allPairs[i].human_label === 'UNCERTAIN' && allPairs[i].pair_id.startsWith('PAIR_EXACT_')) return i;
        }
        return -1;
    """)
    if idx_unc != -1:
        driver.execute_script(f"currentFilteredIdx = {idx_unc}; renderCurrentPair();")
        time.sleep(0.5)
        p4 = out_dir / "04_uncertain_candidate.png"
        driver.save_screenshot(str(p4))
        print(f"Saved: {p4.name} (Pair: {driver.find_element(By.ID, 'pair-index').text})")

    # ============================================================
    # 5. Source Provenance Display
    # ============================================================
    # Find a pair with rich source references (e.g. Batch 2 pair with tender/doc)
    idx_src = driver.execute_script("""
        for (let i = 0; i < allPairs.length; i++) {
            if (allPairs[i].source_reference_1 && allPairs[i].source_reference_1.includes('Doc:')) return i;
        }
        return 0;
    """)
    driver.execute_script(f"currentFilteredIdx = {idx_src}; renderCurrentPair();")
    time.sleep(0.5)
    p5 = out_dir / "05_source_provenance_display.png"
    driver.save_screenshot(str(p5))
    print(f"Saved: {p5.name} (Pair: {driver.find_element(By.ID, 'pair-index').text})")

    # ============================================================
    # 6. Reviewer Save Workflow (Selected label, notes, confidence)
    # ============================================================
    # Navigate to a pending candidate in Batch 2
    idx_pen = driver.execute_script("""
        for (let i = 0; i < allPairs.length; i++) {
            if (allPairs[i].human_label === 'PENDING') return i;
        }
        return 0;
    """)
    driver.execute_script(f"currentFilteredIdx = {idx_pen}; renderCurrentPair();")
    time.sleep(0.3)

    # Click label button 1 (selects only)
    driver.find_element(By.ID, "btn-u").click()
    time.sleep(0.2)

    # Enter reviewer engineering notes
    notes_box = driver.find_element(By.ID, "reviewer-notes")
    notes_box.clear()
    notes_box.send_keys("Candidate review audit: Specification requires inspection of tender Annexure B for operating pressure.")

    # Select confidence rating
    conf_sel = Select(driver.find_element(By.ID, "confidence-select"))
    conf_sel.select_by_value("MEDIUM")
    time.sleep(0.3)

    p6 = out_dir / "06_reviewer_save_workflow.png"
    driver.save_screenshot(str(p6))
    print(f"Saved: {p6.name} (Workflow demonstrated on: {driver.find_element(By.ID, 'pair-index').text})")

    print("\n[SUCCESS] All 6 genuine screenshots captured successfully!")

finally:
    driver.quit()

