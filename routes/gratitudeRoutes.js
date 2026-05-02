const express = require("express");
const router = express.Router();
const Gratitude = require("../models/Gratitude"); // make sure model exists

// POST: Save gratitude
router.post("/api/gratitude", async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    await Gratitude.create({ text });

    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Server error" });
  }
});

module.exports = router;