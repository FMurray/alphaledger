import streamlit as st
import json
import db  # Import the new database module
from pathlib import Path  # For creating cache directories

# Alphaledger imports
from alphaledger.universe import Universe, Security
from alphaledger.config import settings as alphaledger_settings
import polars as pl  # For DataFrame manipulation if needed outside alphaledger
import datetime

# --- Alphaledger Tag Mapping ---
# Based on the TAGS dictionary from xbrl_parse_exp.ipynb
# Maps canonical tags (from JSONL) to a list of possible XBRL concept_names
XBRL_TAG_MAP = {
    "revenue_total": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "NetSales",
    ],
    "revenue_goods": ["SalesRevenueGoodsNet"],
    "revenue_services": ["SalesRevenueServicesNet"],
    "revenue_interest_dividends": ["InterestAndDividendIncomeOperating"],
    "revenue_net_of_interest_expense": ["TotalRevenueNetOfInterestExpense"],
    "cost_of_goods_and_services_sold": ["CostOfGoodsAndServicesSold"],
    "cost_of_revenue": ["CostOfRevenue"],
    "cost_of_goods_sold": ["CostOfGoodsSold"],
    "cogs_excl_d_and_a": [
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss", "OperatingIncome"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_basic_total": ["EarningsPerShareBasic"],
    "eps_basic_cont_ops": ["IncomeLossFromContinuingOperationsPerBasicShare"],
    "eps_diluted_total": ["EarningsPerShareDiluted"],
    "eps_diluted_cont_ops": ["IncomeLossFromContinuingOperationsPerDilutedShare"],
    "research_and_development_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopment",
        "ResearchAndDevelopmentExpenseCredit",
    ],
    "selling_general_admin_expense": ["SellingGeneralAndAdministrativeExpense"],
    "general_admin_expense": ["GeneralAndAdministrativeExpense"],
    "income_tax_expense": ["IncomeTaxExpenseBenefit", "ProvisionForIncomeTaxes"],
    "current_income_tax_expense": ["CurrentIncomeTaxExpenseBenefit"],
    "cash_flow_from_operations": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash_flow_from_investing": ["NetCashProvidedByUsedInInvestingActivities"],
    "cash_flow_from_financing": ["NetCashProvidedByUsedInFinancingActivities"],
    "capital_expenditures": [
        "CapitalExpenditures",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    ],
    "share_repurchases": [
        "RepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfCommonStock",
        "AcceleratedShareRepurchasesSettlementPaymentOrReceipt",
        "StockRepurchasedAndRetiredDuringPeriodValue",
    ],
    "cash_and_cash_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "cash_equiv_and_short_term_investments": [
        "CashCashEquivalentsAndShortTermInvestments"
    ],
    "accounts_receivable_net": ["AccountsReceivableNet"],
    "accounts_and_notes_receivable_net": ["AccountsAndNotesReceivableNet"],
    "accounts_receivable_current": ["AccountsReceivableNetCurrent"],
    "inventory_net": ["InventoryNet"],
    "inventory_finished_goods_wip": ["InventoryFinishedGoodsAndWorkInProcess"],
    "accounts_payable_current": ["AccountsPayableCurrent"],
    "accounts_payable_trade_current": ["AccountsPayableTradeCurrent"],
    "assets_total": ["Assets"],
    "assets_total_net": ["AssetsNet"],
    "assets_current": ["AssetsCurrent"],
    "liabilities_total": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "stockholders_equity": ["StockholdersEquity"],
    "long_term_debt_and_capital_leases": ["LongTermDebtAndCapitalLeaseObligations"],
    "long_term_debt": ["LongTermDebt"],
    "debt_current": ["DebtCurrent"],
    "commercial_paper": ["CommercialPaper"],
    "short_term_borrowings": ["ShortTermBorrowings"],
    "current_portion_long_term_debt": ["CurrentPortionOfLongTermDebt"],
    "current_portion_long_term_debt_alt": ["LongTermDebtCurrent"],
    "finance_lease_liability_current": ["FinanceLeaseLiabilityCurrent"],
    "finance_lease_liability_non_current": ["FinanceLeaseLiabilityNoncurrent"],
    "debt_total": ["Debt"],
    "goodwill": ["Goodwill"],
}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_alphaledger_facts(ticker: str, year: int, cache_buster: str = "v1"):
    st.write(
        f"Fetching Alphaledger facts for {ticker}, FY{year} using default Alphaledger cache (cache_buster: {cache_buster})..."
    )

    facts_df = pl.DataFrame()  # Initialize empty DataFrame
    # Define the columns we absolutely want to ensure are in the output if available
    desired_columns = [
        "concept_name",
        "fact_value",
        "unit",
        "period_instant",
        "period_start",
        "period_end",
        "context_id",
        "context_scenario",
        "ticker",
        "report_date",
        # Add any other essential columns from the full list you provided
    ]

    try:
        universe_name = f"{ticker}_{year}_temp_streamlit_universe"
        temp_universe = Universe(
            universe_name, start_year=year, end_year=year, empty=True
        )
        temp_security = Security(ticker=ticker, name=ticker, exchange="Unknown")
        temp_universe.add_security(temp_security)
        temp_universe.year_range.start = year
        temp_universe.year_range.end = year

        st.write(
            f"Collecting filings metadata for {ticker} FY{year} (using default AL cache)..."
        )
        filings_lf = temp_universe.collect_filings()

        if filings_lf is None or filings_lf.collect().is_empty():
            st.warning(
                f"No filings metadata found by Alphaledger for {ticker} FY{year}."
            )
            return facts_df

        st.write(
            f"Processing numeric facts for {ticker} FY{year} (using default AL cache)..."
        )
        numeric_facts_lf = temp_universe.get_numeric_facts()

        if numeric_facts_lf is not None:
            collected_facts_full = numeric_facts_lf.collect()
            if not collected_facts_full.is_empty():
                # Filter by year first
                filtered_by_year_df = collected_facts_full
                if (
                    "report_date" in collected_facts_full.columns
                    and collected_facts_full["report_date"].dtype == pl.Date
                ):
                    filtered_by_year_df = collected_facts_full.filter(
                        pl.col("report_date").dt.year() == year
                    )
                    st.success(
                        f"Found {len(filtered_by_year_df)} Alphaledger facts for {ticker} FY{year} after year filter."
                    )
                elif "report_date" in collected_facts_full.columns:
                    st.warning(
                        f"Alphaledger 'report_date' column for {ticker} FY{year} is not of Date type: {collected_facts_full['report_date'].dtype}. Year filter might be inaccurate."
                    )
                else:
                    st.warning(
                        f"Alphaledger facts for {ticker} FY{year} missing 'report_date' column. Cannot filter by year."
                    )

                # Select only desired columns to ensure the cached DF has them explicitly
                # And to potentially reduce cache size / complexity
                present_desired_columns = [
                    col for col in desired_columns if col in filtered_by_year_df.columns
                ]
                if present_desired_columns:
                    facts_df = filtered_by_year_df.select(present_desired_columns)
                    if not facts_df.is_empty():
                        st.write(
                            f"Selected {len(facts_df.columns)} columns for final DF."
                        )
                    else:
                        st.info(
                            f"Selected columns resulted in an empty DataFrame for {ticker} FY{year}."
                        )
                else:
                    st.warning(
                        f"None of the desired columns were found in Alphaledger facts for {ticker} FY{year}."
                    )

            else:
                st.info(
                    f"Alphaledger processing yielded no numeric facts for {ticker} FY{year}."
                )
        else:
            st.info(
                f"Alphaledger get_numeric_facts returned None for {ticker} FY{year}."
            )

    except Exception as e:
        st.error(f"Error fetching Alphaledger data for {ticker} FY{year}: {e}")
        import traceback

        st.text(traceback.format_exc())

    return facts_df


def main():
    db.init_db()  # Initialize the database at the start of the app
    st.title("JSONL Annotation Tool (DB Version)")

    # Initialize session state variables
    if "current_view_index" not in st.session_state:  # Index in record_ids_in_view
        st.session_state.current_view_index = 0
    if (
        "record_ids_in_view" not in st.session_state
    ):  # List of source_record.id for current file
        st.session_state.record_ids_in_view = []
    if "uploaded_file_name_cache" not in st.session_state:  # To track current file
        st.session_state.uploaded_file_name_cache = None
    if "chosen_dataset_option" not in st.session_state:
        st.session_state.chosen_dataset_option = (
            None  # Will hold selected file name or "Upload new file"
        )

    # Check for existing datasets in the DB
    available_files_in_db = db.get_distinct_uploaded_file_names()

    # Sidebar option to select existing dataset or upload new
    dataset_options = ["Upload new file"] + available_files_in_db

    # Use a different key for the widget to avoid conflict if we re-assign chosen_dataset_option
    selected_option_from_widget = st.sidebar.selectbox(
        "Choose dataset or upload new",
        options=dataset_options,
        index=0,  # Default to "Upload new file"
        key="dataset_selector_widget",
    )

    # Update chosen_dataset_option based on widget, this helps manage flow
    # especially if the list of available_files_in_db changes after an upload
    if st.session_state.chosen_dataset_option != selected_option_from_widget:
        st.session_state.chosen_dataset_option = selected_option_from_widget
        # If a new dataset is selected from DB (not upload), clear uploader cache to force DB load path
        if st.session_state.chosen_dataset_option != "Upload new file":
            st.session_state.uploaded_file_name_cache = (
                None  # Reset to allow loading from DB
            )
            st.rerun()  # Rerun to process the new selection cleanly

    uploaded_file = None
    if st.session_state.chosen_dataset_option == "Upload new file":
        uploaded_file = st.sidebar.file_uploader("Upload a JSONL file", type=["jsonl"])

    # Determine if we are loading a new file or an existing one from DB
    load_from_db = False
    file_to_process_name = None

    if uploaded_file:
        file_to_process_name = uploaded_file.name
        # This is the new file upload path
        if st.session_state.uploaded_file_name_cache != file_to_process_name:
            st.session_state.uploaded_file_name_cache = file_to_process_name
            # ... (rest of the new file processing logic from before) ...
            # (This will be nested inside the 'if uploaded_file:' block)
            raw_lines = [line for line in uploaded_file]
            with st.spinner(f"Processing {uploaded_file.name}..."):
                for i, line in enumerate(raw_lines):
                    try:
                        record_data = json.loads(line)
                        source_record_id = db.add_source_record(
                            uploaded_file.name, i, record_data
                        )
                        if source_record_id:
                            if "values" in record_data and isinstance(
                                record_data["values"], list
                            ):
                                for j, item_data in enumerate(record_data["values"]):
                                    db.add_value_item(source_record_id, j, item_data)
                    except json.JSONDecodeError:
                        st.error(
                            f"Error decoding JSON on line {i + 1} in {uploaded_file.name}."
                        )
                    except Exception as e:
                        st.error(f"An unexpected error processing line {i + 1}: {e}")

            st.session_state.record_ids_in_view = db.get_record_ids_for_file(
                uploaded_file.name
            )
            st.session_state.current_view_index = 0
            if st.session_state.record_ids_in_view:
                st.success(
                    f"Processed and loaded {len(st.session_state.record_ids_in_view)} records from {uploaded_file.name}."
                )
                # Refresh available_files_in_db in case this was a new one
                st.session_state.chosen_dataset_option = (
                    uploaded_file.name
                )  # Select the newly uploaded file
                st.rerun()  # Rerun to update the selectbox and select the new file
            else:
                st.warning(f"No records loaded from {uploaded_file.name}.")
        # If uploaded_file is not None but name matches cache, we assume it's already processed and selected.
        # The main display logic will handle showing it.

    elif (
        st.session_state.chosen_dataset_option
        and st.session_state.chosen_dataset_option != "Upload new file"
    ):
        # This is the load from DB path
        if (
            st.session_state.uploaded_file_name_cache
            != st.session_state.chosen_dataset_option
        ):
            file_to_process_name = st.session_state.chosen_dataset_option
            st.session_state.uploaded_file_name_cache = file_to_process_name
            st.session_state.record_ids_in_view = db.get_record_ids_for_file(
                file_to_process_name
            )
            st.session_state.current_view_index = 0
            if st.session_state.record_ids_in_view:
                st.success(
                    f"Loaded {len(st.session_state.record_ids_in_view)} records for {file_to_process_name} from database."
                )
            else:
                st.warning(
                    f"No records found in database for {file_to_process_name}, though it was listed."
                )
        else:
            # chosen_dataset_option is an existing file and it matches the cache, so data is ready
            file_to_process_name = st.session_state.uploaded_file_name_cache

    # --- Main display logic starts here, relies on session state being set above ---
    if not st.session_state.record_ids_in_view:
        if (
            st.session_state.chosen_dataset_option == "Upload new file"
            and not uploaded_file
        ):
            st.info("Please upload a JSONL file to begin annotation.")
        elif (
            st.session_state.chosen_dataset_option
            and st.session_state.chosen_dataset_option != "Upload new file"
        ):
            pass  # Message already shown if loading from DB failed or was empty
        else:
            st.info("Select a dataset or upload a new JSONL file to begin annotation.")
        return

    # The rest of your existing main() function for displaying records and annotations
    # Ensure it uses st.session_state.uploaded_file_name_cache for the file name display
    # and st.session_state.record_ids_in_view and st.session_state.current_view_index for navigation

    total_records_in_view = len(st.session_state.record_ids_in_view)
    current_display_file_name = (
        st.session_state.uploaded_file_name_cache
        if st.session_state.uploaded_file_name_cache
        else "No file selected"
    )

    st.sidebar.write(
        f"Record {st.session_state.current_view_index + 1} of {total_records_in_view} (File: {current_display_file_name})"
    )

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Previous") and st.session_state.current_view_index > 0:
        st.session_state.current_view_index -= 1
    if (
        col2.button("Next")
        and st.session_state.current_view_index < total_records_in_view - 1
    ):
        st.session_state.current_view_index += 1

    current_db_record_id = st.session_state.record_ids_in_view[
        st.session_state.current_view_index
    ]

    # Display current record for annotation
    source_record = db.get_source_record_by_id(current_db_record_id)

    if not source_record:
        st.error(
            f"Could not fetch record with ID {current_db_record_id} from the database."
        )
        return

    try:
        # The full_record_json is stored as a string, parse it back to dict for display
        record_content = json.loads(source_record["full_record_json"])
    except json.JSONDecodeError:
        st.error("Failed to parse the stored JSON for the current record.")
        record_content = {}  # Fallback to empty dict

    st.subheader(
        f"Annotating Record: {source_record['ticker']} - FY{source_record['fy']} (DB ID: {source_record['id']})"
    )
    # Display the original full JSON record for context
    with st.expander("View Full Original Record JSON"):
        st.json(record_content)

    # Fetch Alphaledger facts for the current ticker and year
    alphaledger_facts_df = pl.DataFrame()  # Default to empty DF
    if source_record["ticker"] and source_record["fy"]:
        # Ensure FY is an integer for the function call
        try:
            current_fy = int(source_record["fy"])
            with st.spinner(
                f"Loading Alphaledger data for {source_record['ticker']} FY{current_fy}..."
            ):
                # Pass the cache_buster, change its value in code to force a refresh
                alphaledger_facts_df = get_alphaledger_facts(
                    source_record["ticker"], current_fy, cache_buster="v2"
                )
        except ValueError:
            st.error(
                f"Fiscal year {source_record['fy']} is not a valid integer. Cannot fetch Alphaledger data."
            )
        except Exception as e:
            st.error(
                f"An unexpected error occurred while trying to fetch Alphaledger data: {e}"
            )
    else:
        st.info(
            "Ticker or Fiscal Year missing from source record, cannot fetch Alphaledger data."
        )

    # DEBUG: Inspect the DataFrame from Alphaledger
    if not alphaledger_facts_df.is_empty():
        st.subheader("[Debug] Alphaledger DataFrame Details:")
        st.write("Columns:", alphaledger_facts_df.columns)
        st.write("Data Types:", alphaledger_facts_df.dtypes)
        st.dataframe(alphaledger_facts_df.head(10))  # Show top 10 rows
    else:
        st.info("[Debug] Alphaledger DataFrame is empty.")

    value_items_with_annotations = db.get_value_items_for_record(current_db_record_id)

    if value_items_with_annotations:
        for item in value_items_with_annotations:
            st.markdown("---")
            original_jsonl_tag = item["original_tag"]
            st.write(f"**Tag:** {original_jsonl_tag}")
            st.write(
                f"**Original JSONL Value:** {item['original_value']} {(item['original_units'] or '')}"
            )
            if item["original_comments"]:
                st.write(f"**Original JSONL Comments:** {item['original_comments']}")

            # Display Alphaledger facts
            if (
                not alphaledger_facts_df.is_empty()
                and original_jsonl_tag in XBRL_TAG_MAP
            ):
                xbrl_concept_names_to_check = XBRL_TAG_MAP[original_jsonl_tag]
                # Ensure 'concept_name' column exists before filtering
                if "concept_name" in alphaledger_facts_df.columns:
                    matched_alphaledger_facts = alphaledger_facts_df.filter(
                        pl.col("concept_name").is_in(xbrl_concept_names_to_check)
                    )
                    if not matched_alphaledger_facts.is_empty():
                        st.markdown("**Alphaledger Filing Data:**")
                        for al_fact in matched_alphaledger_facts.iter_rows(named=True):
                            fact_display_parts = [
                                f"Concept: `{al_fact.get('concept_name', 'N/A')}`",
                                f"Value: {al_fact.get('fact_value', 'N/A')}",
                                f"Unit: {al_fact.get('unit', 'N/A')}",
                            ]

                            period_info = ""
                            if al_fact.get("period_instant"):
                                period_info = f"Period: {al_fact.get('period_instant')}"
                            elif al_fact.get("period_start") and al_fact.get(
                                "period_end"
                            ):
                                period_info = f"Period: {al_fact.get('period_start')} to {al_fact.get('period_end')}"
                            if period_info:
                                fact_display_parts.append(period_info)

                            # Simplified context display for debugging
                            context_id_val = al_fact.get("context_id", None)
                            context_scenario_val = al_fact.get("context_scenario", None)

                            if context_id_val:
                                fact_display_parts.append(f"CtxID: {context_id_val}")
                            if context_scenario_val:
                                fact_display_parts.append(
                                    f"Scenario: {str(context_scenario_val)}"
                                )

                            st.markdown(f"  - {' | '.join(fact_display_parts)}")
                    else:
                        st.markdown(
                            f"_No matching Alphaledger facts found for concepts: {xbrl_concept_names_to_check}_ (Fiscal Year {source_record['fy']})"
                        )
                else:
                    st.markdown("_Alphaledger data is missing 'concept_name' column._")
            elif original_jsonl_tag not in XBRL_TAG_MAP:
                st.markdown(
                    f"_Tag '{original_jsonl_tag}' not in XBRL_TAG_MAP, cannot look up Alphaledger data._"
                )

            value_item_db_id = item["id"]

            # Pre-fill from DB if annotation exists
            current_annotated_val = (
                item["annotated_value"]
                if item["annotated_value"] is not None
                else str(item["original_value"])
            )
            current_annotation_comment = (
                item["annotation_comment"]
                if item["annotation_comment"] is not None
                else ""
            )
            current_status = item["status"] if item["status"] else "pending"

            st.info(f"Status: {current_status.capitalize()}")

            annotated_value_input = st.text_input(
                f"Annotated Value for Tag: {original_jsonl_tag}",
                value=current_annotated_val,
                key=f"val_input_{value_item_db_id}",
            )
            annotation_comment_input = st.text_area(
                f"Annotation Comment for Tag: {original_jsonl_tag}",
                value=current_annotation_comment,
                key=f"comment_input_{value_item_db_id}",
            )

            action_cols = st.columns(3)
            if action_cols[0].button(
                "Save Annotation", key=f"save_btn_{value_item_db_id}"
            ):
                db.save_annotation(
                    value_item_db_id,
                    annotated_value_input,
                    annotation_comment_input,
                    status="annotated",
                )
                st.success(
                    f"Annotation for '{original_jsonl_tag}' (Item ID: {value_item_db_id}) saved."
                )
                st.rerun()  # Rerun to refresh displayed values and status

            if action_cols[1].button(
                "Mark as Verified", key=f"verify_btn_{value_item_db_id}"
            ):
                # Save current input fields as well, or assume they are already saved if verifying
                db.save_annotation(
                    value_item_db_id,
                    annotated_value_input,
                    annotation_comment_input,
                    status="verified",
                )
                st.success(
                    f"Annotation for '{original_jsonl_tag}' (Item ID: {value_item_db_id}) marked as verified."
                )
                st.rerun()

            if current_status != "pending":
                if action_cols[2].button(
                    "Revert to Pending", key=f"revert_btn_{value_item_db_id}"
                ):
                    # Reverting might mean clearing the annotation text or just changing status
                    # For now, let's keep the text but change status
                    db.save_annotation(
                        value_item_db_id,
                        annotated_value_input,
                        annotation_comment_input,
                        status="pending",
                    )
                    st.info(
                        f"Annotation for '{original_jsonl_tag}' (Item ID: {value_item_db_id}) reverted to pending."
                    )
                    st.rerun()

    else:
        st.write("No 'value items' found for this record in the database.")


if __name__ == "__main__":
    main()
