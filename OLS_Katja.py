

     # Effect on exam score of hours used, controlling for gpa_prior, female and field:
    print()
    print("REGRESSION 10:")

    result4 = analyze_data.regression(
        model="OLS",
        y="exam_score",
        x="hours_used + gpa_prior + female + C(field)"
    )

    analyze_data.save_results_as_text(result4, output_path / "tabel_10.txt")

    # Effect on exam score of assigned, controlling for gpa_prior, female and field:
    print()
    print("REGRESSION 11:")              

    result5 = analyze_data.regression(
        model="OLS",
        y="exam_score",
        x="assigned + gpa_prior + female + C(field)"
    )

    analyze_data.save_results_as_text(result5, output_path / "tabel_11.txt")

    #First stage regression for hours used
    print()
    print("REGRESSION 12:")     

    result6 = analyze_data.regression(
        model="OLS",
        y="hours_used",
        x="assigned + gpa_prior + female + C(field)"
    )
    
    analyze_data.save_results_as_text(result6, output_path / "tabel_12.txt")

    #effect on exam score of assigned, no controls
    print()
    print("REGRESSION 12:")     

    result7 = analyze_data.regression(
        model="OLS",
        y="exam_score",
        x="assigned"
    )
    
    analyze_data.save_results_as_text(result7, output_path / "tabel_17.txt")

   
