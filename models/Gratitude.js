const mongoose = require("mongoose");

const gratitudeSchema = new mongoose.Schema({
  text: String,
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model("Gratitude", gratitudeSchema);