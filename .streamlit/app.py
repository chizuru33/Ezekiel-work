# ============================================================
# CHOLERA SURVEILLANCE SYSTEM - SMS ALERT INTEGRATION
# ============================================================

import streamlit as st
import requests
import numpy as np


# ============================================================
# SMS FUNCTION
# ============================================================

def send_sms(phone_number, message):

    api_key = st.secrets["TERMII_API_KEY"]

    url = "https://api.ng.termii.com/api/sms/send"

    payload = {
        "to": phone_number,
        "from": "CholeraAlert",
        "sms": message,
        "type": "plain",
        "channel": "generic",
        "api_key": api_key
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=15
        )

        if response.status_code == 200:
            return True, response.json()

        return False, response.text

    except Exception as e:

        return False, str(e)


# ============================================================
# PAGE: RISK ASSESSMENT
# ============================================================

elif nav == "🔍 Risk Assessment":

    st.markdown("# 🔍 Real-Time Community Risk Assessor Pipeline")

    st.markdown(
        "Input environmental features to output "
        "projected predictive categories."
    )

    st.markdown("---")


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    (
        trained_models,
        scores,
        scaler,
        le_cats,
        le_risk,
        enc_features,
        X_test_s,
        y_test,
        best_name
    ) = train_model(df_raw)


    best_mdl = trained_models[best_name]


    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("assessment_form"):

        st.markdown(
            "### 📋 Community Environmental Input Form"
        )

        cl1, cl2, cl3 = st.columns(3)


        # ----------------------------------------------------
        # ENVIRONMENTAL VARIABLES
        # ----------------------------------------------------

        with cl1:

            rf = st.slider(
                "Rainfall (mm)",
                0.0,
                400.0,
                50.0
            )

            tp = st.slider(
                "Temperature (°C)",
                15.0,
                45.0,
                31.0
            )

            hm = st.slider(
                "Humidity (%)",
                10.0,
                100.0,
                45.0
            )


        # ----------------------------------------------------
        # COMMUNITY VARIABLES
        # ----------------------------------------------------

        with cl2:

            ta = st.slider(
                "Household Toilet Access (%)",
                0.0,
                100.0,
                40.0
            )

            od = st.slider(
                "Open Defecation (%)",
                0.0,
                100.0,
                25.0
            )

            pop = st.number_input(
                "Population",
                min_value=1,
                value=50000
            )


        # ----------------------------------------------------
        # HEALTH INFRASTRUCTURE
        # ----------------------------------------------------

        with cl3:

            hfc = st.number_input(
                "Health Facilities",
                min_value=0,
                value=15
            )

            hosp = st.number_input(
                "Hospitals / Clinics",
                min_value=0,
                value=2
            )

            hw = st.number_input(
                "Health Workers",
                min_value=0,
                value=30
            )

            elev = st.number_input(
                "Elevation (meters)",
                value=400
            )


        # ----------------------------------------------------
        # WASH PARAMETERS
        # ----------------------------------------------------

        st.markdown("### 🌐 WASH Parameters")

        cx1, cx2, cx3, cx4 = st.columns(4)


        with cx1:

            pws = st.selectbox(
                "Primary Water Source",
                WATER_SOURCES
            )


        with cx2:

            stype = st.selectbox(
                "Sanitation Type",
                SANITATION_TYPES
            )


        with cx3:

            frl = st.selectbox(
                "Flood Risk",
                ["Low", "High"]
            )


        with cx4:

            alvl = st.selectbox(
                "Access Level",
                ["Low", "Medium", "High"]
            )


        # ----------------------------------------------------
        # SMS INFORMATION
        # ----------------------------------------------------

        st.markdown("### 📱 SMS Alert Information")

        community_name = st.text_input(
            "Community / LGA Name",
            placeholder="Example: Bauchi LGA"
        )

        phone_number = st.text_input(
            "Health Authority Phone Number",
            placeholder="Example: 2348012345678"
        )


        # ----------------------------------------------------
        # RUN PREDICTION
        # ----------------------------------------------------

        evaluate = st.form_submit_button(
            "⚡ Run Risk Evaluation"
        )


    # ========================================================
    # PROCESS PREDICTION
    # ========================================================

    if evaluate:

        # ----------------------------------------------------
        # ENCODE CATEGORICAL VARIABLES
        # ----------------------------------------------------

        w_enc = le_cats[
            "Primary_Water_Source"
        ].transform([pws])[0]

        s_enc = le_cats[
            "Sanitation_Type"
        ].transform([stype])[0]

        f_enc = le_cats[
            "Flood_Risk_Level"
        ].transform([frl])[0]

        a_enc = le_cats[
            "Access_Level"
        ].transform([alvl])[0]


        # ----------------------------------------------------
        # CREATE INPUT VECTOR
        # ----------------------------------------------------

        input_vector = np.array([[
            rf,
            tp,
            hm,
            ta,
            od,
            hfc,
            hosp,
            hw,
            elev,
            pop,
            w_enc,
            s_enc,
            f_enc,
            a_enc
        ]])


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        scaled_vec = scaler.transform(
            input_vector
        )


        # ----------------------------------------------------
        # MACHINE LEARNING PREDICTION
        # ----------------------------------------------------

        pred_idx = best_mdl.predict(
            scaled_vec
        )[0]


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        proba_list = best_mdl.predict_proba(
            scaled_vec
        )[0]


        # ----------------------------------------------------
        # CONVERT PREDICTION TO RISK LABEL
        # ----------------------------------------------------

        risk_output = le_risk.inverse_transform(
            [pred_idx]
        )[0]


        confidence = (
            proba_list[pred_idx] * 100
        )


        # ----------------------------------------------------
        # DISPLAY RISK
        # ----------------------------------------------------

        color = RISK_COLORS.get(
            risk_output,
            "#0066cc"
        )


        st.markdown(
            f"""
            <div style="
                background:{color}15;
                border:3px solid {color};
                padding:20px;
                border-radius:8px;
            ">

                <h3 style="
                    color:{color};
                    margin:0;
                ">
                    RISK LEVEL:
                    {risk_output.upper()}
                </h3>

                <p style="color:#333;">
                    Prediction Confidence:
                    <strong>
                        {confidence:.2f}%
                    </strong>
                </p>

                <p style="color:#333;">
                    Community:
                    <strong>
                        {community_name if community_name else "Not specified"}
                    </strong>
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # HIGH-RISK SMS ALERT
        # ====================================================

        if risk_output.lower() == "high":

            st.warning(
                "⚠️ HIGH ENVIRONMENTAL RISK DETECTED. "
                "An early-warning SMS can be sent to the "
                "designated health authority."
            )


            # ------------------------------------------------
            # SEND SMS BUTTON
            # ------------------------------------------------

            if st.button(
                "📲 Send SMS Alert"
            ):

                if not community_name:

                    st.error(
                        "Please enter the "
                        "Community / LGA name."
                    )


                elif not phone_number:

                    st.error(
                        "Please enter the "
                        "health authority phone number."
                    )


                else:

                    # ----------------------------------------
                    # SMS MESSAGE
                    # ----------------------------------------

                    message = (
                        "CHOLERA EARLY WARNING: "
                        f"High environmental risk detected "
                        f"in {community_name}. "
                        f"Risk confidence is "
                        f"{confidence:.1f}%. "
                        "Please investigate and take "
                        "appropriate public-health action."
                    )


                    # ----------------------------------------
                    # SEND SMS
                    # ----------------------------------------

                    success, result = send_sms(
                        phone_number,
                        message
                    )


                    if success:

                        st.success(
                            "✅ SMS alert sent successfully."
                        )

                        st.info(
                            f"Alert sent to: {phone_number}"
                        )

                    else:

                        st.error(
                            "❌ SMS could not be sent."
                        )

                        st.code(
                            str(result)
                        )


        # ====================================================
        # LOW / MEDIUM RISK
        # ====================================================

        else:

            st.info(
                "ℹ️ No high-risk SMS alert is required "
                "for this prediction."
            )
