import { fetch } from "./http";

export interface User {
  id: number;
  email: string;
}

export type UserId = User["id"];

export enum Role {
  Admin,
  Member,
}

/** Formats a value. */
export function format(value: string): string;
export function format(value: number): string;
export function format(value: string | number): string {
  return String(value);
}

export class UserService {
  private cache = new Map<UserId, User>();

  constructor(private readonly baseUrl: string) {}

  async getUser(id: UserId): Promise<User> {
    const parse = (raw: string) => JSON.parse(raw) as User;
    return parse(await fetch(`${this.baseUrl}/users/${id}`));
  }
}

export const debounce = (fn: () => void, ms: number) => {
  let t: ReturnType<typeof setTimeout>;
  return () => {
    clearTimeout(t);
    t = setTimeout(fn, ms);
  };
};
