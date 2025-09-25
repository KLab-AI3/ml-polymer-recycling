import React from "react";
import "../static/style.css";

const Header: React.FC = () => {
  return (
    <header className="header header--academic" role="banner">
      <div className="header__container">
        <h1 className="header__title" tabIndex={0}>
          AI Polymer Classification <span className="header__subtitle">(Raman &amp; FTIR)</span>
        </h1>
        <p className="header__desc">
          AI-driven polymer aging prediction and classification using spectroscopy and deep learning.
        </p>
      </div>
    </header>
  );
};

export default Header;
