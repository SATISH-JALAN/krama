import "./Citations.css";

export default function Citations({ ids }: { ids: string[] }) {
  if (ids.length === 0) return null;
  return (
    <div className="citations">
      <span className="eyebrow">cited</span>
      {ids.map((id) => (
        <span key={id} className="citation-chip mono">
          {id}
        </span>
      ))}
    </div>
  );
}
