import { useState } from "react";
import type { AppConfig } from "../api/types";

/** Always-visible notice that age estimation is probabilistic. */
export function Notice({ config }: { config: AppConfig }) {
  const [open, setOpen] = useState(false);
  return (
    <aside className="notice" role="note" aria-label="About age estimation">
      <div className="notice-row">
        <span className="notice-icon" aria-hidden>
          !
        </span>
        <p>
          <strong>Age estimation is probabilistic.</strong> SafeFrame can be wrong — anyone it is
          unsure about is blurred by default. Always review before sharing.{" "}
          <button className="link" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            {open ? "Less" : "More"}
          </button>
        </p>
      </div>
      {open && (
        <div className="notice-more">
          <p>{config.notice}</p>
          <p>{config.intended_use}</p>
        </div>
      )}
    </aside>
  );
}
