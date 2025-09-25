import React from "react";
import "../static/style.css";

const Header: React.FC = () => {
  return (
    <div className="header">
      <h1 className="header__title">
        AI Polymer Classification
        <span className="header__subtitle">(Raman & FTIR)</span>
      </h1>
      <p className="header__desc">
        AI-driven polymer aging prediction and classification using spectroscopy
        and deep learning.
      </p>
    </div>
  );
};

export default Header;
