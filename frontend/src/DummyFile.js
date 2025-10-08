import React from "react";
import Dashboard from "./dashboard";

const obj = {
  sentiment: {
    "+": 42,   // positive comments count
    "-": 16,   // negative comments count
    "0": 25    // neutral comments count
  },
  wordcount: {
    usability: 61,
    users: 54,
    form: 45,
    tool: 33,
    good: 30,
    free: 28,
    reviews: 26,
    testing: 24,
    website: 22,
    text: 20,
    forms: 18,
    product: 18,
    review: 16,
    information: 16,
    policy: 15,
    input: 13,
    design: 13,
    feedback: 13,
    need: 12,
    testingtool: 12,
    easy: 10,
    allow: 10,
    best: 10,
    new: 9,
    option: 8,
    important: 7,
    one: 6,
    questions: 6,
    required: 6,
    example: 5,
    clear: 5,
    support: 5,
    help: 4,
    user: 4,
    allows: 4,
    allowsreview: 3,
    analytics: 3,
    settings: 3,
    walkthrough: 3,
    valid: 3,
    scenario: 3,
    fine: 2,
    ask: 2,
    capturing: 2
  },
  important_rare: [
    "Some feedback forms are too long, users drop off before finishing.",
    "The help section doesn't answer all usability questions.",
    "Forms should allow multi-language support for broader reach.",
    "Common errors while submitting forms aren't clearly explained.",
    "Review analytics should be available for all admin users.",
    "Data privacy policy lacks information on form data retention.",
    "Website should provide clear walkthrough for new users.",
    "Product reviews are sometimes flagged incorrectly by the tool.",
    "Testing workflow needs an option for anonymous input.",
    "Accessibility investigation is pending for the mobile version."
  ]
};
const DummyData = ()=>{
  return(
    <Dashboard obj={obj}/>
  )
}
export default DummyData;
