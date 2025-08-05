export const formatNumber = (num: number | string): string => {
  if (num === null || num === undefined) return "-";
  const parsed = Number(num);
  if (isNaN(parsed)) return "-";
  return parsed.toLocaleString("en-IN");
};
